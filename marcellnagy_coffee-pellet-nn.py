fastDebug = False

tmpImageFormat = 'bmp'

upScaleToPreventErrors = 1000
!pip install pytorch-lightning pandas
def maxRect(lhs, rhs):

    if lhs[2] == -1:

        result = rhs

    elif rhs[2] == -1:

        result = lhs

    else:

        left = min(lhs[0], rhs[0])

        top = min(lhs[1], rhs[1])

        lhsRight = lhs[0] + lhs[2]

        lhsBottom = lhs[1] + lhs[3]

        rhsRight = rhs[0] + rhs[2]

        rhsBottom = rhs[1] + rhs[3]

        right = max(lhsRight, rhsRight)

        bottom = max(lhsBottom, rhsBottom)

        result = (left, top, right - left + 1, bottom - top + 1)

    return result



def getBBoxOfContent(frame):

    result = (0, 0, -1, -1)

    dilationSize = 3

    element = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (2 * dilationSize + 1, 2 * dilationSize + 1),

        (dilationSize, dilationSize)

    )

    frame = cv2.dilate(frame, element)

    _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        poly = cv2.approxPolyDP(contour, 3, True)

        result = maxRect(result, cv2.boundingRect(poly))

    return result
import numpy as np

import pandas as pd

from glob import glob



measurementEntities = []



for filename in glob('/kaggle/input/**/*.txt', recursive = True):

    path = filename[:-4]

    logData = pd.read_table(filename, dtype = {'Name': str, 'Mass (mg)': np.float32})

    for entity in zip(logData['Name'], logData['Mass (mg)']):

        measurementEntities.append({'fileName': entity[0], 'filePath': path + '/' + entity[0] + '.mp4', 'mass': entity[1]})



if fastDebug:

    measurementEntities = sorted(measurementEntities, key = lambda e: e['mass'])

    measurementEntities = measurementEntities[0:10]

    

print('Count of handled videos:', len(measurementEntities))
import cv2

import matplotlib.pyplot as plt

import shutil

import os.path

from ipywidgets import IntProgress

from IPython.display import display



allImagesCount = 0

predicatedMasses = []



statVideoPixels = []

statVideoMasses = []



print('Removing existing images...')

if os.path.exists('/tmp'):

    shutil.rmtree('/tmp')

    os.mkdir('/tmp')



print('Splitting videos into small images...')

progressBar = IntProgress(min = 0, max = len(measurementEntities))

display(progressBar)



fig = plt.figure(figsize=(10,15))

fig.tight_layout()

statTransformedFramePlot = fig.add_subplot(311)

statOriginalFramePlot = fig.add_subplot(312)

statOriginalVideoPlot = fig.add_subplot(313)



for entity in measurementEntities:

    reader = cv2.VideoCapture(entity['filePath'])

    totalMass = entity['mass']

    totalUsedPixels = 0.0

    videoData = []

    maxBBox = (0, 0, -1, -1)

    

    statFramePixels = []

    statFrameTransformedMasses = []

    statFrameOriginalMasses = []

    

    while (True):

        success, frame = reader.read()

        if not success:

            break

        

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        _, frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)

        

        contentBBox = getBBoxOfContent(frame)

        maxBBox = maxRect(maxBBox, contentBBox)

        

        usedPixels = frame.sum() / 255

        totalUsedPixels += usedPixels

        

        statFramePixels.append(usedPixels)

        

        videoData.append({'fileName': entity['fileName'], 'frame': frame[:], 'usedPixels': usedPixels})

    

    statVideoPixels.append(totalUsedPixels)

    statVideoMasses.append(totalMass)

    

    boxLeft, boxTop, boxWidth, boxHeight = maxBBox

        

    for i, frameData in enumerate(videoData):

        predicatedFullFrameMass = frameData['usedPixels'] / totalUsedPixels * totalMass

        frame = frameData['frame']

        fileName = frameData['fileName']

        

        usedFullFramePixels = frame.sum() / 255

        

        statFrameOriginalMasses.append(predicatedFullFrameMass)

        

        # If size < 64, important details are lost due to resizing.

        size = 64

        

        frameHeight, frameWidth = frame.shape

        dropHeight = frameHeight // 4 * 3

        scale = boxWidth / frameWidth

        scaledHeight = frameHeight * scale

        height = int(min(scaledHeight, dropHeight))

        top = max(0, frameHeight - height)

        

        frame = frame[top : top + height, boxLeft : boxLeft + boxWidth]

        

        frame = cv2.resize(frame, (size, size))

        

        usedSubFramePixels = frame.sum() / 255

        if usedSubFramePixels == 0.0:

            statFrameTransformedMasses.append(0.0)

            continue

        

        predicatedOnePixelMass = predicatedFullFrameMass / usedFullFramePixels

        predicatedSubFrameMass = predicatedOnePixelMass * usedSubFramePixels

        

        predicatedSubFrameMass *= upScaleToPreventErrors

        predicatedMasses.append(predicatedSubFrameMass)

        

        statFrameTransformedMasses.append(predicatedSubFrameMass)

        

        newImageFileName = f'/tmp/{fileName:s}-{i:d}-{predicatedSubFrameMass:.20f}.{tmpImageFormat:s}'

        cv2.imwrite(newImageFileName, frame)

        allImagesCount += 1

    

    statOriginalFramePlot.scatter(statFramePixels, statFrameOriginalMasses)

    statTransformedFramePlot.scatter(statFramePixels, statFrameTransformedMasses)

    

    progressBar.value += 1



massStdDev = np.std(predicatedMasses)

massMean = np.mean(predicatedMasses)



statOriginalFramePlot.set_title('Predicated mass per brightness of original frames')

statTransformedFramePlot.set_title('Predicated mass per brightness of resized/cropped frames')

statOriginalVideoPlot.set_title('Original mass per brightness of all frames of the video')

statOriginalVideoPlot.scatter(statVideoPixels[1:], statVideoMasses[1:])



plt.show()
from torch.utils.data import Dataset, DataLoader

import time



def getImageList(fromIndex = None, toIndex = None):

    return glob('/tmp/*.' + tmpImageFormat)[fromIndex : toIndex]



print('Count of used images:', len(getImageList()))



class PelletMassMeasurementDataset(Dataset):

    def __init__(self, fromIndex, toIndex):

        self.data = []

        files = getImageList(fromIndex, toIndex)

        for fileName in files:

            mass = fileName.split('-')[-1]

            mass = mass.split('.' + tmpImageFormat)[0]

            mass = np.float32(mass)

            self.data.append({'fileName': fileName, 'mass': mass})



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        data = self.data[idx]

        

        image = cv2.imread(data['fileName'], cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32)

        image /= 255.0

        

        image = np.expand_dims(image, axis = 0)

        

        return [image, data['mass']]
chunks = allImagesCount // 10

trainEnd = chunks * 8

validationEnd = allImagesCount



print('Bounds of dataset parts:', 0, trainEnd, validationEnd)



pelletTrainDataLoader = DataLoader(

    PelletMassMeasurementDataset(0, trainEnd),

    batch_size = 1,

    shuffle = True,

    num_workers = 16

)

pelletValidationDataLoader = DataLoader(

    PelletMassMeasurementDataset(trainEnd, validationEnd),

    batch_size = 1,

    shuffle = False,

    num_workers = 16

)
import torch

import torch.nn as nn



class Detector(nn.Module):

    def __init__(self, batchSize = 1):

        super(Detector,self).__init__()

        # Similar to nn.Linear, but works with a single number.

        self.scaler = nn.Parameter(torch.randn(1, requires_grad = True))

        self.bias = nn.Parameter(torch.randn(1, requires_grad = True))

        

    def forward(self, x):

        result = x.sum() * self.scaler + self.bias

        result = result.abs()

        result = (result - massMean) / massStdDev

        return result
import torch.optim as optim

import pytorch_lightning as pl



cudaEnabled = torch.cuda.is_available()

print('Cuda enabled:', cudaEnabled)



class PelletSystem(pl.LightningModule):

    def __init__(self):

        super(PelletSystem, self).__init__()



        torch.manual_seed(42)

        torch.cuda.manual_seed(42)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        

        self.trainingAccuracy = []

        self.validationAccuracy = []

        

        self.net = Detector()

        if cudaEnabled:

            self.net = self.net.cuda()



        self.criterion = nn.SmoothL1Loss()

        if cudaEnabled:

            self.criterion = self.criterion.cuda()



    def configure_optimizers(self):

        return optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)



    @pl.data_loader

    def train_dataloader(self):

        return pelletTrainDataLoader



    @pl.data_loader

    def val_dataloader(self):

        return pelletValidationDataLoader

    

    

    def forward(self, x):

        return self.net.forward(x)



    def _appendMean(self, outputs, array, key):

        lossMean = torch.stack([x['loss'] for x in outputs]).mean()

        array.append(lossMean.item())

        return {key: lossMean}

    

    def _evaluate(self, batch):

        x, y = batch

        if cudaEnabled:

            x, y = x.cuda(), y.cuda()

        predicated = self.forward(x)

        loss = self.criterion(predicated, y)

        return {'loss': loss}

    

    

    def training_step(self, batch, batch_idx):

        return self._evaluate(batch)

    

    def training_epoch_end(self, outputs):

        return self._appendMean(outputs, self.trainingAccuracy, 'train_loss')



    

    def validation_step(self, batch, batch_idx):

        return self._evaluate(batch)

      

    def validation_epoch_end(self, outputs):

        return self._appendMean(outputs, self.validationAccuracy, 'val_loss')
model = PelletSystem()



trainer = pl.Trainer(max_epochs = 10)

trainer.fit(model)
print(model.net.scaler.item())

print(model.net.bias.item())



plt.plot(model.trainingAccuracy)

plt.plot(model.validationAccuracy)

plt.show()
def predicatePelletVideoMass(fileName):

    predicatedMass = 0.0



    reader = cv2.VideoCapture(testVideo)

    reader.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while (True):

        success, frame = reader.read()

        if not success:

            break

        

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        _, frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)

        

        frame = np.expand_dims(frame, axis = 0)

        frame = np.expand_dims(frame, axis = 0)

        frame = torch.Tensor(frame)

        

        if cudaEnabled:

            frame = frame.cuda()



        output = model(frame).cpu().item()

        output = output / 2 + (output / 100) ** 1.6

        predicatedMass += output



    predicatedMass /= upScaleToPreventErrors * 100 * 2

    

    return predicatedMass
measurementFile = glob('/kaggle/input/**/short.txt', recursive = True)[0]

measurementLog = pd.read_table(measurementFile, dtype = {'Name': str, 'Mass (mg)': np.float32})



testCount = min(len(measurementEntities), 118)



totalDifferences = []



for index in range(testCount):

    testVideo = measurementFile[:-4] + '/' + measurementLog['Name'][index] + '.mp4'

    testMass = measurementLog['Mass (mg)'][index]

    

    predicatedMass = predicatePelletVideoMass(testVideo)

    predicatedMass = abs(predicatedMass)

    predicatedMass = round(predicatedMass, 4)



    difference = round(testMass - predicatedMass, 4)

    totalDifferences.append(abs(difference))

    print('Video #' + str(index), '- Original:', testMass, 'Predicated:', predicatedMass, 'Difference', difference)

    

print('Total difference:', np.sum(totalDifferences) / testCount)



plt.plot(totalDifferences)

plt.show()



plt.scatter(measurementLog['Mass (mg)'][0 : testCount], totalDifferences)

plt.show()