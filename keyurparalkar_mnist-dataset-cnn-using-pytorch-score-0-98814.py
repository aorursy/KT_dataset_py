%reload_ext autoreload
%autoreload 2
%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from PIL import Image
PATH = '../input/'
def return_data_label(fileName):
    df = pd.read_csv(filepath_or_buffer=PATH+fileName)
    if(fileName != 'test.csv'):
        label = np.array(df['label'])
        data = np.array(df[df.columns[1:]],dtype=np.float)
        new_data = np.reshape(a=data,newshape=(data.shape[0],28,28))
        return new_data, label
    else:
        data = np.array(df,dtype=np.float)
        new_data = np.reshape(a=data,newshape=(data.shape[0],28,28))
        return new_data
        
trainData, trainLabel = return_data_label('train.csv')
testData = return_data_label('test.csv')
#preprocessing the dataset
trainData = trainData / 255
trainData = (trainData - 0.5)/0.5

testData = testData / 255
testData = (testData - 0.5)/0.5

trainData = torch.from_numpy(trainData)
testData = torch.from_numpy(testData)
trainData, testData = trainData.type(torch.FloatTensor), testData.type(torch.FloatTensor)
trainData = trainData.unsqueeze_(dim=1)
testData = testData.unsqueeze_(dim=1)
trainDataset = torch.utils.data.TensorDataset(trainData,torch.from_numpy(trainLabel))
trainDataLoader = torch.utils.data.DataLoader(trainDataset,batch_size=100,shuffle=False, num_workers=4)

# testDataset = torch.utils.data.TensorDataset(testData)
testDataLoader = torch.utils.data.DataLoader(testData,batch_size=100,shuffle=False, num_workers = 4)
print("Training batches == \n",len(trainData))
print("Testing batches == \n",len(testData))

#visualizing no. of examples of each type
def total_count(loader):
    totalClassCount = [0,0,0,0,0,0,0,0,0,0]

    for batch_id,(images,labels) in enumerate(loader):
        for label in labels:
            totalClassCount[int(label)] += 1
    return totalClassCount
classes = [0,1,2,3,4,5,6,7,8,9]
print("Digit class = ",classes)
totalCount = total_count(trainDataLoader)

fig0, ax0 = plt.subplots()
ax0.barh(y=classes,width=totalCount)
ax0.set_xlabel('# Examples')
ax0.set_ylabel('# Digit Classes')
ax0.set_title('Train Set')

#Visualizing single digit:
temp = trainDataLoader.dataset[0][0].numpy()
temp = np.reshape(a=temp,newshape=(temp.shape[1],temp.shape[2]))
plt.imshow(temp)
#Creating LeNet5 nn class module
# conv2d => relu => maxpooling => conv2d => relu => maxpooling => fully connected layer(fc)1 
#=> fc2 => softmax output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(1024,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.fc1Size = 0
        self.toKnowMaxPoolSize= False
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),1)
        
        if(self.toKnowMaxPoolSize == True):
            self.fc1Size = x.size()
            print(x.size())
            return
        #now lets reshape the matrix i.e. unrolling the matrix
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
n1 = Net()
n1 = n1.cuda()
print(n1)

#Creating function for training our NNs
def train_model(model,mode,decay,criterion,dataloader,optimizer,dictionary,num_epochs=30):
    #mode = True means model.train() and False is model.eval()
    #decay = True means decrease LR with no. of epochs 
    
    totalLoss = []
    totalLRs = []
    correct = 0
    total = 0
    LR = 0
    
    for epoch in range(num_epochs):
        if(decay == True):
            for param in optimizer.param_groups:
                LR = param['lr'] * (0.1**(epoch//7))
                param['lr'] = LR
            totalLRs.append(LR)
            
        print("Epoch = {}/{} ".format(epoch,num_epochs),end=" ")
        for batch_id,(image, label) in enumerate(dataloader):
            if(mode == True):
                optimizer.zero_grad()
                image = torch.autograd.Variable(image)
                label = torch.autograd.Variable(label)
                image = image.cuda()
                label = label.cuda()
            else:
                image = torch.autograd.Variable(image)
                image = image.cuda()

            output = model.forward(image)
            
            if(mode == True):
                loss = criterion(output,label)

            _, predictated = torch.max(output.data,1)
            
            if(mode == True):
                correct += (predictated == label.data).sum()
                total += label.size(0)

                loss.backward()
                optimizer.step()

            del image,label
            
        torch.cuda.empty_cache()
        print("Loss = {:.5f}".format(loss.data[0]))
        totalLoss.append(loss.data[0])
        
    dictionary['totalLoss'] = totalLoss
    dictionary['correct'] = correct
    dictionary['totalSize'] = total
    dictionary['totalLRs'] = totalLRs
    
    return model,dictionary
# forward => loss => backward => upadte weights
n1.toKnowMaxPoolSize = False   # To print the size of the last maxpool layer.
optimizer = torch.optim.SGD(n1.parameters(),lr=0.1)
criterion = nn.CrossEntropyLoss().cuda()
#Let's first find correct Learning Rate using Learning decay.

dictModel = {}
n1, dictModel = train_model(model=n1,mode=True,decay=True,criterion=criterion,dataloader=trainDataLoader,optimizer=optimizer,dictionary=dictModel,num_epochs=50)
# LOSS vs EPOCHS
plt.plot(dictModel['totalLoss'])
# Loss vs LR 
plt.plot(dictModel['totalLRs'],dictModel['totalLoss'])
print("Accuracy == ",100*(dictModel['correct']/dictModel['totalSize']))
# forward => loss => backward => upadte weights
n1.toKnowMaxPoolSize = False     # To print the size of the last maxpool layer.
optimizer = torch.optim.SGD(n1.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss().cuda()

dictModel = {}
n1, dictModel = train_model(model=n1,mode=True,decay=False,criterion=criterion,dataloader=trainDataLoader,optimizer=optimizer,dictionary=dictModel,num_epochs=20)
plt.plot(dictModel['totalLoss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Dataset')
#Accuracy
print("Accuracy == ",100*(dictModel['correct']/dictModel['totalSize']))
#evaluating test set
# forward => loss => backward => upadte weights
n1.toKnowMaxPoolSize = False     # To print the size of the last maxpool layer.
avgLossTest = []
totalPrediction = []

for batch_id,image in enumerate(testDataLoader):
        image = torch.autograd.Variable(image)
        image = image.cuda()

        output = n1.forward(image)
        
        _, predictated = torch.max(output.data,1)
        
        totalPrediction.append(predictated)
temp = [list(x.cpu().numpy()) for x in totalPrediction]
Label = []

for x in temp:
    for y in x:
        Label.append(y)
# with open('res.txt','r+') as fp:
#     fp.write(str(temp))
ImageId = [t for t in range(1,28001)]
len(ImageId)
subData = {
    'ImageId':ImageId,
    'Label':Label
}
df = pd.DataFrame(data=subData)
df.to_csv(path_or_buf='submission.csv',index=False)
