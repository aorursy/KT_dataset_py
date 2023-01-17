import numpy as np

import pandas as pd

import os

from PIL import Image

from sklearn.utils import shuffle

from glob import glob

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

from torchvision import models

from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F



from time import time

import matplotlib.pyplot as plt

from sklearn import metrics

import pickle as pkl
# dirs

WORKING_DIR = '/kaggle/working/'

INPUT_DIR = '/kaggle/input/data/'



# classes

imgClasses = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']



# Use GPU is available

if torch.cuda.is_available():

  device = torch.device('cuda:0')

else:

  device = torch.device('cpu')
# load up dataframe

df = pd.read_csv(f'{INPUT_DIR}Data_Entry_2017.csv')



# keep only unique findings

df = df[df['Finding Labels'].map(lambda x: True if len(x.split('|')) == 1 else False)]



# helper function to extract what we need

def findingExtractor(df, targetClass = imgClasses):

  for i in targetClass:

    tdf = df[df['Finding Labels'].map(lambda x: True if x == i else False)]



    if i == targetClass[0]:

      rdf = tdf

    else:

      rdf = pd.concat([rdf, tdf])



  rdf = shuffle(rdf, random_state = 69).reset_index(drop = True)



  return rdf



df = findingExtractor(df)



# dataframe helper

def processDF(df, topLabels = imgClasses):

    # create a field which houses full path to the images

    allImagesGlob = glob(f'{INPUT_DIR}images*/images/*.png')

    allImagesPathDict = {os.path.basename(x): x for x in allImagesGlob}

    df['path'] = df['Image Index'].map(allImagesPathDict.get)

    

        

    for label in topLabels:

        df[label] = df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)

        

    df['finalLabel'] = df.apply(lambda x: [x[topLabels].values], 1).map(lambda x: x[0])



    # topLables

    tempdf = df['Finding Labels']



    # drop dups

    # data.drop_duplicates(subset ="Patient ID", keep = False, inplace = True) 

        

    # drop not req columns

    dropLabels = ['Finding Labels', 'Image Index', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11']

    df.drop(columns = dropLabels, inplace = True)

    df.drop(columns = topLabels, inplace = True)



    # add

    df = pd.concat([df, tempdf], axis = 1)

    

    return df



# get final df

df = processDF(df)



# func that calculate class weights

def getWeights(df, trainClass = imgClasses):

    weights = []

    for i in trainClass:

        t = len(df[df['Finding Labels'] == i]) / len(df)

        weights.append(t)

        

    return weights

        

classWeights = getWeights(df)
# dataset loader class



# split dataset into train and test

trainset, testset = train_test_split(df, shuffle = True, test_size=0.1, random_state = 69)

testset, valset = train_test_split(testset, shuffle = True, test_size=0.5, random_state = 69)





class DataGenerator(Dataset):

    def __init__(self, df, batch_size=8,  trainClass = imgClasses, dim=(224, 224), shuffle=True, to_fit=True, classesNo = len(imgClasses), transformations = None):

        self.itemList = df.values.tolist()

        self.trainClass = trainClass

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.dim = dim

        self.to_fit = to_fit

        self.classesNo = classesNo

        self.transformations = transformations

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.itemList) / self.batch_size))



    def __getitem__(self, index):

        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]



        # Find list of IDs

        itemListTemp = [self.itemList[k] for k in indexes]



        # Generate data

        X = self._generate_X(itemListTemp)



        if self.to_fit:

            y = self._generate_y(itemListTemp)

            return X, y

        else:

            return X



    def on_epoch_end(self):

        self.indexes = np.arange(len(self.itemList))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def _generate_X(self, itemListTemp):

      for i in itemListTemp:

        originalImage = Image.open(i[0]).convert('RGB') # open up

        resizedImage = originalImage.resize(self.dim) # resize

        arrayImage = np.array(resizedImage) # array-fy

        arrayImage = self.transformations(arrayImage) # apply transformations

        

        if i == itemListTemp[0]:

          imgs = arrayImage.view(-1, 3, 224, 224)

        else:

          imgs = torch.cat((imgs, arrayImage.view(-1, 3, 224, 224)), axis = 0)



      return imgs

        



    def _generate_y(self, itemListTemp):

      for i in itemListTemp:

        # t = np.array(i[1]).astype('int')



#         onehotarr = np.eye(self.classesNo)

        t = np.array([self.trainClass.index(i[2])])



        if i == itemListTemp[0]:

          labels = t#[np.newaxis, :]

        else:

          labels = np.concatenate((labels, t), axis = 0)



      return torch.from_numpy(labels).long()

# define Model class

class CheXnet(nn.Module):

	def __init__(self, out_size):

		super(CheXnet, self).__init__()

		self.densenet121 = torchvision.models.densenet121(pretrained=True)

		num_ftrs = self.densenet121.classifier.in_features

		self.densenet121.classifier = nn.Sequential(

		    nn.Linear(num_ftrs, out_size)

		)



	def forward(self, x):

		x = self.densenet121(x)

		return x, F.log_softmax(x, 1)

    



# load model

model = CheXnet(len(imgClasses)).to(device)

optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))

scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 2, mode = 'min', min_lr = 1e-8)



classWeights = torch.FloatTensor(getWeights(df)).to(device)

lossFunc = nn.CrossEntropyLoss(classWeights)



# load tained weights

# load model

## init model class to gpu

model = CheXnet(len(imgClasses)).to(device)

optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))

scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 2, mode = 'min', min_lr = 1e-8)



## load wghts

modelCheckpoint = torch.load(f'/kaggle/input/chexnet-trained-30/1591530786-chesXnet-train.tar')

model.load_state_dict(modelCheckpoint['stateDict'])

optimizer.load_state_dict(modelCheckpoint['optimizerStateDict'])
# training

epochs = 30

batch_size = 16

verbose = True



# transformations

transformations = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(

    224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# Use to Store Losses for iters

lossListTrain = []

lossListTest = []



# acc list

accListTrain = []

accListTest = []



# variable to store iters

iterNo = 0



# datagens

trainGen = DataGenerator(trainset, batch_size, transformations=transformations)

valGen = DataGenerator(valset, batch_size, transformations=transformations)



for epoch in range(epochs):

    loss = 0

    acc = 0

    for idxTrain in range(trainGen.__len__()):

        # zero Model grads and put model in train mode

        model.train()

        model.zero_grad()



        # loading true imgs

        X, y = trainGen.__getitem__(idxTrain)

        X = X.to(device)

        y = y.to(device)



        # train

        output = model(X)[0]

        error = lossFunc(output, y)

        error.backward()



        # loss and acc

        loss += error.item()

        predOH = torch.max(output, 1)[1]

        predT = y

        acc += (predOH == predT).float().sum()



        # optimize grads

        optimizer.step()



    lossListTrain.append(loss / trainGen.__len__())

    accListTrain.append(acc / trainGen.__len__() / batch_size)



    if ((epoch + 1) % 2) == 0 or epoch == 0:

        # test

        loss = 0

        acc = 0

        for idxTest in range(valGen.__len__()):

            # put Model in eval mode

            model.eval()



            # loading true imgs

            X, y = valGen.__getitem__(idxTest)

            X = X.to(device)

            y = y.to(device)



            # train

            output = model(X)[0]

            error = lossFunc(output, y)



            # loss and acc

            loss += error.item()

            predOH = torch.max(output, 1)[1]

            predT = y

            acc += (predOH == predT).float().sum()



        lossListTest.append(loss / valGen.__len__())

        accListTest.append(acc / valGen.__len__() / batch_size)

        

        # shuffle generator

        valGen.on_epoch_end()

        

    else:

        lossListTest.append(lossListTest[-1])

        accListTest.append(accListTest[-1])



    # shuffle generator

    trainGen.on_epoch_end()



    # reduce lr

    scheduler.step(lossListTest[-1])



    if ((epoch + 1) % 6) == 0:

        try:

            print('Chking the model!')

            # save final models after taining ends

            torch.save(model.state_dict(),

                       f'{WORKING_DIR}{int(time())}-chesXnet.pt')

            print('Done!!!')

            print('')

        except:

            print("An exception occurred")



    print(f'[{epoch + 1} / {epochs}]')

    print(f'Train Loss: {lossListTrain[-1]:.5f}  Test loss: {lossListTest[-1]:.5f}')

    print(f'Train Acc: {accListTrain[-1]:.5f}  Test Acc: {accListTest[-1]:.5f}')

    print('')



try:

    print('Saving the model1!')

    torch.save({

        'stateDict': model.state_dict(),

        'optimizerStateDict': optimizer.state_dict()

    }, f'{WORKING_DIR}{int(time())}-chesXnet-train.tar')

    print('Done!!!')

    print('')

except:

    print("An exception occurred")
# Plot Lossed

plt.figure(figsize=(10,5))

plt.title('Acc and Loss')

plt.plot(lossListTrain,label="Train Loss")

plt.plot(lossListTest,label="Test Loss")

plt.plot(accListTrain,label="Train Acc")

plt.plot(accListTest,label="Test Acc")

plt.xlabel("iters")

plt.legend()

plt.show()
# # test the model

# testGen = DataGenerator(testset, batch_size = 100, transformations = transformations)

# Xtest, ytest = testGen.__getitem__(0)



# # get results

# with torch.no_grad():

#   Xtest = Xtest.to(device)

#   pred = model(Xtest)[1]



# ytest = ytest.numpy()

# pred = pred.cpu().numpy()



# # Print the confusion matrix

# print(metrics.confusion_matrix(ytest, np.argmax(pred, axis = 1)))



# # Print the precision and recall, among other metrics

# print(metrics.classification_report(ytest, np.argmax(pred, axis = 1)))