#basic imports

%matplotlib inline



import sys

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image



import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor') #all tensors use the GPU



from torchvision.models import vgg16

from torch.nn import Module, Sequential, MSELoss, L1Loss

from torch.autograd import Variable

import torch.nn.functional as F



#activation to transform input images between 0 and 1 to VGG expected input

def caffeActivation(x):

    x = 255 * x

    x1 = x[:,2:,:,:] - 103.939

    x2 = x[:,1:2,:,:] - 116.779

    x3 = x[:,:1,:,:] - 123.68



    return torch.cat([x1,x2,x3], 1)



#reshapes loaded images to pytorch convolutional networks

def torchFormat(x):

    #creates batch size dimension and move channels to dim 1

    reshaped = np.moveaxis(np.expand_dims(x, 0), -1,1)

    return Variable(torch.tensor(reshaped, dtype=torch.float), requires_grad=False)



#reshapes images from torch for plotting

def plotFormat(x):

    #moves channels to last dimension

    return np.moveaxis(x, 1, -1)[0]



#opens an image and returns it resized and in range between 0 and 1

def openImage(file, size):

    return np.asarray(Image.open(file).resize(resizeSize,Image.LANCZOS))/255.



#a plotting function

def plotImages(columns,*images, **kwargs):

    #kwargs handling

    rowHeight = kwargs.pop('rowHeight',6)

    writeValues = kwargs.pop('writeValues', False)

    titles = kwargs.pop('titles', None)

    bottomTexts = kwargs.pop('bottomTexts',None)

    

    if len(kwargs) > 0:

        warnings.warn('plotImages got some invalid arguments: ' + str([key for key in kwargs]))

    

        

    #actual function

    n = len(images)

    rowsDiv = divmod(n,columns)

    rows = rowsDiv[0]

    if rowsDiv[1] > 0: rows+=1

    

    fig, ax = plt.subplots(ncols=columns, nrows=rows,squeeze=False, figsize=(16,rowHeight * rows))

    

    hspace=0

    counter = Incrementer2D(columns)

    for i in range(n):

        image = images[i]

        if image.max() > 1:

            image = image/255.

        ax[counter.i,counter.j].imshow(image,cmap=cm.Greys_r if len(image.shape) == 2 else None)

        ax[counter.i,counter.j].get_xaxis().set_ticks([])

        ax[counter.i,counter.j].get_yaxis().set_ticks([])

        

        if writeValues:

            for (y,x),label in np.ndenumerate(image):

                txt = str(label)[:5] if len(str(label)) >= 5 else str(label)

                ax[counter.i,counter.j].text(x,y,txt,ha='center',va='center',color='blue')

        

        if not titles is None:

            ax[counter.i,counter.j].set_title(titles[i])

            hspace+=.06

        

        if not bottomTexts is None:

            ax[counter.i,counter.j].annotate(bottomTexts[i], (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points', va='top')   

            hspace+=.06

        

        counter.increment()

            

    

    for i in range(n,columns*rows):

        ax[counter.i,counter.j].axis('off')

        counter.increment()

        

    if hspace > 0: plt.subplots_adjust(hspace=hspace/(rowHeight*rows))

    plt.show()

    

#helper for the plotting function

class Incrementer2D:

    def __init__(self, limitForJ, startI = 0, startJ = 0):

        self.limitForJ = limitForJ

        self.i = startI

        self.j = startJ

        

    def increment(self):

        self.j += 1

        if self.j == self.limitForJ:

            self.j = 0

            self.i += 1

            

print("A hidden image plotting function is also defined here")
def gramMatrix(imageTensor):

    #shape and pixels

    shp = list(imageTensor.size()) #shape

    pixels = shp[2] * shp[3]

    

    #flattening the spatial dimensions, keeping channels and batch

    original = imageTensor.view(shp[0], shp[1], pixels)

    

    #normalization of the final result - to keep the same order at the end

    original = torch.sqrt(original / pixels)

    

    #gram matrix

    transposed = original.permute(0,2,1)    

    return torch.matmul(original, transposed)



#loss to be used with gram matrices

def gramLoss(yPred, yTrue):

#     #yPred = features extracted from trainable input

#     #yTrue = features extracted from the style image

    

    #the loss from gram matrices:

    trueGram = gramMatrix(yTrue)

    predGram = gramMatrix(yPred)

    

    return L1Loss()(predGram,trueGram)

class VGGFeatureExtractor(Module):

    def __init__(self, vggModel, layerIndices):

        super(VGGFeatureExtractor,self).__init__()

        

        #get the sequential convolutional part of the VGG

        #each model is built differently, so you'd need different approaches

        self.features = vggModel.features

        

        #desired layers

        self.layerIndices = layerIndices

        self.outputCount = len(layerIndices)

        

        #making this model's parameters untrainble

        for p in self.parameters():

            p.requires_grad = False

        

    #outputs from the selected layers

    def forward(self,x):

        outputs = list()

        

        #for each layer in the VGG

        for i, layer in enumerate(self.features.children()):

            x = layer(x) #apply the layer

            

            #if this layer is a desired layer, store its outputs

            if i in self.layerIndices:

                outputs.append(x)

                

                #check if we got all desired layers

                if i == self.layerIndices[-1]:

                    return outputs

                    

        return outputs
class StyleTransferModel(object):

    

    def __init__(self,imageShape,inputActivation,baseFeatureExtractor,styleFeatureExtractor,

                 baseWeights = 1,styleWeights = 1):

        super(StyleTransferModel, self).__init__()

        

        #shape

        self.imageShape = imageShape

        self.inputActivation = inputActivation 

        self.batchShape = (1,) + imageShape

        

        #feature extraction models

        self.baseModel = baseFeatureExtractor

        self.styleModel = styleFeatureExtractor

        

        #output feature counts

        self.baseOutputCount = baseFeatureExtractor.outputCount

        self.styleOutputCount = styleFeatureExtractor.outputCount

        

        #weights applied to each selected layer

        self.baseWeights = [baseWeights]* self.baseOutputCount

        self.styleWeights = [styleWeights]* self.styleOutputCount

        self.allWeights = self.baseWeights + self.styleWeights

        



        #preparing loss lists - one loss per output feature 

        self.baseLosses = [L1Loss()]*self.baseOutputCount

        self.styleLosses = [gramLoss]*self.styleOutputCount

        self.allLosses = self.baseLosses + self.styleLosses

        

        #trainable image input

        imageInputTensor = torch.rand(self.batchShape) #random noise image

        self.imageInput = Variable(imageInputTensor, requires_grad=True) #trainable input



    #get features from the trainable input

    def forward_preds(self):

        

        activatedImage = self.inputActivation(self.imageInput)

        

        #trainable predictions:

        basePredictions = []

        stylePredictions = []

        if self.baseOutputCount > 0:

            basePredictions = self.baseModel(activatedImage)

        if self.styleOutputCount > 0:

            stylePredictions = self.styleModel(activatedImage)



        

        outputs = basePredictions + stylePredictions

        return outputs

        



    #calculates losses between trainable input and features from true images

    def forward_loss(self, trainableFeatures, trueFeatures):

        

        #getting losses for each of the selected outputs

        resultLosses = []

        for trainable, true, w, lossFunction in zip(trainableFeatures, trueFeatures, 

                                                    self.allWeights, self.allLosses):

            loss = w*lossFunction(trainable,true)

            resultLosses.append(loss)



        #summing all losses

        finalLoss = resultLosses[0]

        for l in resultLosses[1:]:

            finalLoss = finalLoss + l

        return finalLoss

        

    #gets the trainable input image

    def getInputImage(self):

        return plotFormat(self.imageInput.data.cpu().numpy())

    

    #set the trainable input image

    def setInputImage(self,image):

        image = torchFormat(image)

        self.imageInput.data = torch.Tensor(image)

    



    def fit(self,baseImage,styleImage, epochs, optimizer, lr, patience = 30, bestLoss = sys.float_info.max, verbose=1):

        

        #activated images

        baseImage = self.inputActivation(torchFormat(baseImage))

        styleImage = self.inputActivation(torchFormat(styleImage))

        

        #static values for base and style image features extracted from vgg

        baseImageFeatures = self.baseModel.forward(baseImage) 

        styleImageFeatures = self.styleModel.forward(styleImage)

        trueFeatures = baseImageFeatures + styleImageFeatures        

        

        #the optimizer will train only the input image, nothing else

        optimizer = optimizer([self.imageInput], lr=lr)

        

        if verbose > 0:

            print('called fit, inital img, lr:', lr)

            self.plot()

        

        #train

        patienceCounter = 0

        for e in range(epochs):

            #standard training

            optimizer.zero_grad()

            loss = self.forward_loss(self.forward_preds(), trueFeatures)

            loss.backward()

            optimizer.step()

            

            #since we want our image between 0 and 1, let's clam it

            self.imageInput.data.clamp_(min=0,max=1)

            

            #store best loss and checks if we should interrupt training early

            currLoss = loss.data.item()

            if currLoss < bestLoss:

                bestLoss  = currLoss

                patienceCounter = 0

            else:

                patienceCounter += 1

                if patienceCounter > patience:

                    break

            

            if verbose > 0:

                if e % 200 == 0:

                    print('loss:', loss)

                    self.plot()

                    

    def fitManyLRs(self, baseImage, styleImage, epochList, lrList):

        for epochs, lr in zip(epochList, lrList):

            self.fit(baseImage, styleImg, epochs=epochs, 

                     optimizer=torch.optim.Adam, lr=lr, verbose=0)

    

    

    #plots the trainable image

    def plot(self):

        image = self.getInputImage()

        print(image.min(),image.max())

        plotImages(1,image)

    
imgSize = 500

resizeSize = (imgSize, imgSize)

inShape = (3,imgSize,imgSize)

vgg = vgg16(pretrained=True)



#for base layers, we're taking a few avoiding too many at the beginning:

baseExtractorForCat = VGGFeatureExtractor(vgg, [3,11,13,15,18,20,22,27,29])

baseExtractorForCity = VGGFeatureExtractor(vgg,[3,6,8,13,15,18,20,22,27,29])



#for style layers, let's take all of them (you can try different combinations)

#styleExtractor = VGGFeatureExtractor(vgg, list(range(1,31)))

styleExtractor = VGGFeatureExtractor(vgg, [3,6,8,11,13,15,18,20,22,25,27,29])



#style transfer model, the styleWeights = 50 seems to give good results

styleTransferModel = StyleTransferModel(inShape,caffeActivation, 

                                        baseExtractorForCat,styleExtractor,

                                        baseWeights = 1,styleWeights = 20)#13000000)

#cat face as base

baseImage = openImage('../input/CatFace.jpg', imgSize)

print("base image:")

plotImages(1, baseImage)



#lots of styles to test

styleFiles = ['StarryNight.jpg', 'TwistedTreePainting.jpg', 'GoldenBalls2.jpg', 

              'RandomPainting.jpg', 'ThunderBlueDense.jpg', 'FieldPainting.jpeg', 'OGrito.jpg']

styleImages = [openImage('../input/' + img, imgSize) for img in styleFiles]



#for each style image, train:

for file, styleImg, in zip(styleFiles, styleImages):

    print('training for ' + file)

    

    #setting base image

    styleTransferModel.setInputImage(baseImage)

    

    #training with different learning rates

    styleTransferModel.fitManyLRs(baseImage,styleImg,epochList=[100, 100], lrList=[0.1, 0.01])



    

    #results

    plotImages(2, styleImg, styleTransferModel.getInputImage())

    

    
#a city as base

baseImage = openImage('../input/City2.jpg', imgSize)

print("base image:")

plotImages(1, baseImage)





#lots of styles to test

styleFiles = ['StarryNight.jpg', 'TwistedTreePainting.jpg', 'BrusselsPainting.jpeg',

              'RandomPainting.jpg', 'ThunderBlueDense.jpg', 'FieldPainting.jpeg', 'OGrito.jpg']

styleImages = [openImage('../input/' + img, imgSize) for img in styleFiles]

styleWeights = [10,20,35]#[3200000,6000000,20000000]





#for each style image, train:

for file, styleImg, in zip(styleFiles, styleImages):

    print('training for ' + file)

    

    #for each style weight

    for w in styleWeights:

        print("style weight = " + str(w))

        

        #new model for this weight

        styleTransferModel = StyleTransferModel(inShape,caffeActivation, 

                                    baseExtractorForCity,styleExtractor,

                                    baseWeights = 1,styleWeights = w)

        

        #fitting from random image

        styleTransferModel.fitManyLRs(baseImage,styleImg,epochList=[100, 100], lrList=[0.1, 0.01])

        imageFromRandom = styleTransferModel.getInputImage()

    

        #fitting from base image

        #setting base image

        styleTransferModel.setInputImage(baseImage)

        styleTransferModel.fitManyLRs(baseImage,styleImg,epochList=[100, 100], lrList=[0.1, 0.01])

        imageFromBase = styleTransferModel.getInputImage()



        #results

        print('style image / random initialization / base initialization')

        plotImages(3, styleImg, imageFromRandom, imageFromBase)

    