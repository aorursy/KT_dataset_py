#### import pandas as pd

import numpy as np

import imageio

import os

import matplotlib.pyplot as plt

%matplotlib inline



# TRAINING DATA

imageData=np.array([])

imageY=[]

for curShape in ['circle','square','star','triangle']:

    print("CurShape is {0}".format(curShape))

    for curFile in os.listdir("../input/four-shapes/shapes/{0}".format(curShape))[0:500]:

        if(imageData.shape[0] ==0):

            imageData=imageio.imread("../input/four-shapes/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1)

        else:

            imageData=np.append(imageData,imageio.imread("../input/four-shapes/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1),axis=0)

        imageY.append(curShape)

imageData=imageData.reshape(-1,1,200,200).astype(np.float32)

imageY=np.array(imageY)        

imageY=[[1,0,0,0] if x=='circle' else [0,1,0,0] if x=='square' else [0,0,1,0] if x=='star' else [0,0,0,1] for x in imageY]

imageYLabels=[0 if x=='circle' else 1 if x=='square' else 2 if x=='star' else 3 for x in imageY]

imageY=np.array(imageY).astype(np.float32)

print("Final imageData shape is {0} and that of imageY is {1}".format(imageData.shape,imageY.shape)) # 200,200



# TEST DATA

imageTestData=np.array([])

imageTestY=[]

for curShape in ['circle','square','star','triangle']:

    print("CurShape is {0}".format(curShape))

    for curFile in os.listdir("../input/four-shapes/shapes/{0}".format(curShape))[500:600]:

        if(imageTestData.shape[0] ==0):

            imageTestData=imageio.imread("../input/four-shapes/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1)

        else:

            imageTestData=np.append(imageTestData,imageio.imread("../input/four-shapes/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1),axis=0)

        imageTestY.append(curShape)

imageTestData=imageTestData.reshape(-1,1,200,200).astype(np.float32)

imageTestY=np.array(imageTestY)

imageTestY=[[1,0,0,0] if x=='circle' else [0,1,0,0] if x=='square' else [0,0,1,0] if x=='star' else [0,0,0,1] for x in imageTestY]

imageTestYLabels=[0 if x=='circle' else 1 if x=='square' else 2 if x=='star' else 3 for x in imageTestY]

imageTestY=np.array(imageTestY).astype(np.float32)

print("Final imageTestData shape is {0} and that of imageTestY is {1}".format(imageTestData.shape,imageTestY.shape)) # 200,200
####################################

### MODEL ###

####################################

import torch

import random

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from collections import OrderedDict 



class Flatten(nn.Module):

    def forward(self, input):

        return input.view(-1, 9 * 9 * 1)



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.SequentialModel1 = nn.Sequential(OrderedDict([

        ('conv1' , nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=0)),

        ('relu1' , nn.ReLU()),

        ('pool1' , nn.MaxPool2d(kernel_size=2,stride=2,padding=0)),

        ('conv2' , nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1,padding=0)),

        ('relu2' , nn.ReLU()),

        ('pool2' , nn.MaxPool2d(kernel_size=2,stride=2,padding=0)),

        ('conv3' , nn.Conv2d(in_channels=8,out_channels=4,kernel_size=4,stride=1,padding=0)),

        ('relu3' , nn.ReLU()),

        ('pool3' , nn.MaxPool2d(kernel_size=2,stride=2,padding=0)),

        ('conv4' , nn.Conv2d(in_channels=4,out_channels=1,kernel_size=4,stride=1,padding=0)),

        ('relu4' , nn.ReLU()),

        ('pool4' , nn.MaxPool2d(kernel_size=2,stride=2,padding=0)),

        ('flatten1' , Flatten()),

        ('fc1' , nn.Linear(9*9*1, 8)),

        ('relu5' , nn.ReLU()),

        ('fc2' , nn.Linear(8, 4)),

        ('smax' , nn.Softmax())

        ]))

    

    def forward(self, x):

        #print("Forward Function Entry Size : {0}".format(x.size()))

        x=self.SequentialModel1(x)

       

        return x



model=Net()

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss()

epochRange=300

batches=8

batchSize=8



# Choice List

choiceBatches=[]

for curBatch in range(batches):

    choiceBatches.append(random.choices(list(range(imageData.shape[0])), k=batchSize))



import random

for epoch in range(epochRange):

    totalLoss=0

    for curBatch in range(batches):

        choiceList=choiceBatches[curBatch]

        inputs=Variable(torch.from_numpy(imageData[choiceList,:,:,:]))

        #print("Size of inputs is {0}".format(inputs.size()))

        labels=Variable(torch.from_numpy(np.array(imageY)[[choiceList]]))

        # zero the parameter gradients

        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = model(inputs)

        #print("Before entering the loss criterion outputs={0} and labels={1}".format(outputs.size(),labels.size()))

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # print statistics

        totalLoss += loss.item()

    if(epoch % 10==0):

        print("Epoch : {0} Total Loss : {1}".format(epoch,totalLoss))

        

# Prediction

model.zero_grad()

predictions=model(Variable(torch.from_numpy(imageTestData)))



# Prediction Metrics

def getData(x):

    y=[0,0,0,0]

    y[np.argmax(x)]=1

    return y



predictions=[getData(x) for x in predictions.detach()]

incorrectCount=0

# Getting the accuracy

for x in range(len(predictions)):

    if((imageTestY[x]!= predictions[x]).any()):

        incorrectCount=incorrectCount+1

        

print("Accuracy is {0}".format(1-(incorrectCount/len(predictions))))



# Backing up the model

torch.save(model,"GBPR_FourImages")

# GUIDED BACK PROP RELU

from torch.autograd import Function



class GetGradients():

    def __init__(self, model, gradLayer, stopLayer):

        self.model = model

        self.gradLayer = gradLayer

        self.stopLayer = stopLayer

        self.gradient=None



    def captureGradientThroughBackwardHook(self, layer, grad_input, grad_output):

        print("Entered backward hook")

        print(grad_input[1].size())

        print(grad_input[2].size())

        print(len(grad_output))

        self.finalInputGradients=grad_output[0]

        

    def captureGradientThroughRegisterHook(self, gradients):

        #print("Layer is {0} grad_input size is {1} and grad_output size is {2}".format(layer,grad_input.size(),grad_output.size()))

        self.finalInputGradients=gradients

        

    def getBackwardInputGradients(self):

        return(self.finalInputGradients)



    def __call__(self, x):

        #model._modules['SequentialModel1']._modules[self.gradLayer].register_backward_hook(self.captureGradient)

        layerOutput=None

        for name, module in model._modules['SequentialModel1']._modules.items():

            x = module(x)

            if name == self.gradLayer:

                x.register_hook(self.captureGradientThroughRegisterHook)

                layerOutput=x

            if name == self.stopLayer:

                break

        return layerOutput, x.view(x.size()[0],-1)



class GuidedBackPropReLUFunc(Function):

    def forward(self,input):

        # This is the function that is called when the output value is calculated based on the input data

        mask=(input > 0).type_as(input)

        output=torch.mul(input,mask)

        # What the below command will do is save the data in the graph element when it is being traversed

        # We can then re-use it during backpropagation

        # Generally the input is saved, but we will save the output too

        self.save_for_backward(input, output)

        return(output)

    

    def backward(self,output):

        input,output=self.saved_tensors

        mask1=(input > 0).type_as(output)

        mask2=(output > 0).type_as(output)

        input=input + torch.mul(torch.mul(output,mask1),mask2)

        return(input)



class GuidedBackPropReLUModel:

    def __init__(self,model,gradLayer,stopLayer):

        self.model=model

        self.model.eval()

        self.gradLayer=gradLayer

        self.stopLayer=stopLayer

        self.getGradients = GetGradients(self.model,self.gradLayer,self.stopLayer)

        # For all the RELU layers, we will replace them with GuidedBackPropReLUFuncs

        for name, module in self.model._modules['SequentialModel1']._modules.items():

            if module.__class__.__name__ == 'ReLU':

                self.model._modules['SequentialModel1']._modules[name] = GuidedBackPropReLUFunc()

                

    def forward(self,input):

        return(self.model(input))

    

    def __call__(self,input):

        gradLayerActivations,stopLayerOutput = self.getGradients(input)

        #stopLayerOutput=self.forward(input)

        print("Size of stopLayerOutput is {0}".format(stopLayerOutput.size()))

        # Getting the index which saw the maximum activation

        index = np.argmax(stopLayerOutput.detach().numpy())

        # Recreate the Last Layer with only that index active

        one_hot = np.zeros((1, stopLayerOutput.size()[-1]), dtype = np.float32)

        one_hot[0][index] = 1

        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)

        one_hot = torch.sum(one_hot * stopLayerOutput)

        # Releasing all the accumulated Grads

        #self.model.SequentialModel1.zero_grad()

        # Backward propagation from the layer

        one_hot.backward(retain_graph=True)

        # Recreated Input

        recreatedInput=self.getGradients.getBackwardInputGradients()

        return(recreatedInput)



# Test Data 1 : First Image

model=torch.load("GBPR_FourImages")

model.zero_grad()



gradCamTestData=imageTestData[0].reshape(1,1,200,200)

gradCamTestDataTorch=Variable(torch.from_numpy(gradCamTestData))



guidedBackPropRelu = GuidedBackPropReLUModel(model,"conv1","fc1")

gbMask = guidedBackPropRelu(gradCamTestDataTorch)

    

plt.imshow(imageTestData[0].reshape(200,200))

plt.show()

# Now for all elements of the first layer gradients, we will do averaging over all the channels

gbMask=gbMask[0]

gbMask=gbMask.detach().numpy()

gbMask=np.sum(gbMask,axis=0)

gbMask = cv2.resize(gbMask, (200, 200))

gbMask = gbMask - np.min(gbMask)

gbMask = gbMask / np.max(gbMask)

plt.imshow(gbMask)

plt.show()

recreatedImage=np.multiply(gbMask,imageTestData[0].reshape(200,200))

plt.imshow(recreatedImage)

plt.show()
# Test Data 2 : Last Image



model=torch.load("GBPR_FourImages")

model.zero_grad()



gradCamTestData=imageTestData[-1].reshape(1,1,200,200)

gradCamTestDataTorch=Variable(torch.from_numpy(gradCamTestData))



guidedBackPropRelu = GuidedBackPropReLUModel(model,"conv1","fc1")

gbMask = guidedBackPropRelu(gradCamTestDataTorch)

    

plt.imshow(imageTestData[-1].reshape(200,200))

plt.show()

# Now for all elements of the first layer gradients, we will do averaging over all the channels

gbMask=gbMask[0]

gbMask=gbMask.detach().numpy()

gbMask=np.sum(gbMask,axis=0)

gbMask = cv2.resize(gbMask, (200, 200))

gbMask = gbMask - np.min(gbMask)

gbMask = gbMask / np.max(gbMask)

plt.imshow(gbMask)

plt.show()

recreatedImage=np.multiply(gbMask,imageTestData[-1].reshape(200,200))

plt.imshow(recreatedImage)

plt.show()