# Importing Modules
import os
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn   
from torch.optim import Adam,SGD
from matplotlib import pyplot as plt
import pandas as pd
# Cudnn for internal optimization
torch.backends.cudnn.benchmark = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device
# Dataset Path
path = r'../input/digit-recognizer'
outputPath = r'../input/output/'
# Reading the Data
train = pd.read_csv(os.path.join(path,'train.csv'))
test = pd.read_csv(os.path.join(path,'test.csv'))
sample_submission = pd.read_csv(os.path.join(path,'sample_submission.csv'))
# The label field is for label of the data
# The other fields are pixel values ranging from 0 to 255
train.head()
test.head()
sample_submission.head()
trainY = train.pop('label')
testY = pd.get_dummies(sample_submission.Label)
trainX = train.copy()
testX = test.copy()
trainX
trainY
testY
testX
testY
# Load the model and the dataset in the GPU(if any) , will help to speed up
class datasetClass(Dataset):
    
    def __init__(self,):
        super(datasetClass,self).__init__()
        self.trainX = torch.tensor(trainX.values).type(torch.float32).div(255).sub_(0.1307).div_(0.3081).contiguous().view(-1,28,28).to(device)
        self.trainY = torch.tensor(trainY.values).type(torch.long).to(device)

    def __getitem__(self,index):
        return self.trainX[index],self.trainY[index]
    
    def __len__(self,):
        return len(self.trainY)
datasetClassObj = datasetClass()
iterObj= iter(datasetClassObj)
x,y = next(iterObj)
x.shape
class datasetClassValid(Dataset):
    
    def __init__(self,):
        super(datasetClassValid,self).__init__()
        self.testX = torch.tensor(testX.values).type(torch.float32).div(255).sub_(0.1307).div_(0.3081).contiguous().view(-1,28,28).to(device)
        self.testY = torch.tensor(testY.values).type(torch.long).to(device)
        
    def __getitem__(self,index):
        return self.testX[index],self.testY[index]
    
    def __len__(self,):
        return len(self.testY)
datasetClassValidObj = datasetClassValid()
iterObj= iter(datasetClassValidObj)
x,y = next(iterObj)
x.shape
class modelClass(nn.Module):
    def __init__(self,inputDim,outputDim):
        super(modelClass,self).__init__()

        self.conv2dLayer1 = nn.Conv2d(1,100,kernel_size=5)
        self.conv2dLayer2 = nn.Conv2d(100,1000,kernel_size=2)
        self.conv2dLayer3 = nn.Conv2d(1000,100,kernel_size=2)
        self.conv2dLayer4 = nn.Conv2d(100,100,kernel_size=2)
        self.conv2dLayer5 = nn.Conv2d(100,20,kernel_size=2)
        
        self.batchnorm2dLayer1 = nn.BatchNorm2d(100)
        self.batchnorm2dLayer2 = nn.BatchNorm2d(1000)
        self.batchnorm2dLayer3 = nn.BatchNorm2d(100)
        self.batchnorm2dLayer4 = nn.BatchNorm2d(100)
        self.batchnorm2dLayer5 = nn.BatchNorm2d(20)
        
        self.maxpoolLayer1 = nn.AdaptiveMaxPool2d((16,16))
        self.maxpoolLayer2 = nn.AdaptiveMaxPool2d((8,8))
        self.maxpoolLayer3 = nn.AdaptiveAvgPool2d((4,4))
        self.maxpoolLayer4 = nn.AdaptiveAvgPool2d((4,4))
        self.maxpoolLayer5 = nn.AdaptiveAvgPool2d((2,2))
        
        self.ReLULayer1 = nn.ReLU()
        self.ReLULayer2 = nn.ReLU()
        self.ReLULayer3 = nn.ReLU() 
        self.ReLULayer4 = nn.ReLU()
        self.ReLULayer5 = nn.ReLU() 
        
        self.conv2DropoutLayer1 = nn.Dropout2d()
        
        self.linearLayer1 = nn.Linear(80,40)
        self.linearLayer2 = nn.Linear(40, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self,x):
        x = x.unsqueeze(0).permute(1,0,2,3)
        
        x = self.conv2dLayer1(x)
        x = self.batchnorm2dLayer1(x)
        x = self.ReLULayer1(x)
        x = self.maxpoolLayer1(x)     
        
        x = self.conv2dLayer2(x)
        x = self.batchnorm2dLayer2(x)
        x = self.ReLULayer2(x)
        x = self.maxpoolLayer2(x)
        
        x = self.conv2dLayer3(x)
        x = self.batchnorm2dLayer3(x)
        x = self.ReLULayer3(x)
        x = self.maxpoolLayer3(x)
        
        x = self.conv2DropoutLayer1(x)
        
        x = self.conv2dLayer4(x)
        x = self.batchnorm2dLayer4(x)
        x = self.ReLULayer4(x)
        x = self.maxpoolLayer4(x)
        
        x = self.conv2dLayer5(x)
        x = self.batchnorm2dLayer5(x)
        x = self.ReLULayer5(x)
        x = self.maxpoolLayer5(x)
        
        x = x.view(-1,80)
        x = self.linearLayer1(x)
        x = self.linearLayer2(x)
        
        x = self.logsoftmax(x)
        return x
bs = 128
epochs = 100
Model = modelClass(784,10)
Model.to(device)
dataloaderTrain = DataLoader(datasetClassObj,batch_size=bs,shuffle=True)
optimizer =  SGD(Model.parameters(),lr=0.01)
lossFunc = nn.NLLLoss()
loss  =  torch.tensor(0.1,requires_grad=True)
x,y = next(iter(dataloaderTrain))
x.shape
Model(x).shape
losses = []
for epoch in range(epochs):
    # This step is very important
    Model.train()
    for x,y in dataloaderTrain:
        ydash = Model(x) 
        loss = lossFunc(ydash,y)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()
    losses.append(loss)
    print(f'The epoch {epoch} the Loss is {loss}')

# Loss Graph
cpuLoss = [float(loss.to('cpu')) for loss in losses] 
plt.plot(cpuLoss)
dataloaderValidObj = DataLoader(datasetClassValidObj,batch_size=64,shuffle=False)
plt.imshow(datasetClassValidObj.testX[0].to('cpu').view(28,28))
submitframe = pd.DataFrame(columns=['Label'])
for x,y in dataloaderValidObj:
    Model.eval()
    #plt.imshow(x.to('cpu').view(28,28))
    prediction = torch.argmax(Model(x),dim=1).to('cpu').numpy()
    submitframe = submitframe.append(pd.DataFrame(prediction,columns=['Label']))

submitframe.head()
submitframe.shape
submitframe.to_csv('sample_submission_Result.csv')