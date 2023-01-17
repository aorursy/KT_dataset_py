#importing modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import os
%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
PATH = '../input'
def mean_std(temp):
    totalMean = []
    totalStd = []
    meanL = 0
    stdL = 0
    
    for batch_id,(image,label) in enumerate(temp):
        img = image.numpy()
        meanL = np.mean(img,axis=(0,1,2))
        stdL = np.std(img,axis=(0,1,2))
        totalMean.append(meanL)
        totalStd.append(stdL)
        
    return totalMean,totalStd
transform = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.39099613,0.39099613,0.39099613], std = [0.1970754,0.1970754,0.1970754])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.39099613,0.39099613,0.39099613], std = [0.1970754,0.1970754,0.1970754])
    ])
}
#using torchvision ImageFolder for importing this dataset
trainData = torchvision.datasets.ImageFolder(root=PATH+'/training',transform=transform['train'])
trainDataLoader = DataLoader(trainData,batch_size=4,shuffle=False,num_workers=4)
valData = torchvision.datasets.ImageFolder(root=PATH+'/validation',transform=transform['val'])
valDataLoader = DataLoader(valData,batch_size=4,shuffle=False,num_workers=4)
#Load the types of classes
trainDataLoader.dataset.classes
# # For calculating mean and std for each image so as to put this in normalization.
# # This cell is to be runned with out trainDataLoader.dataset transforms.Normalize turned off
# x, y = mean_std(trainDataLoader.dataset)
# x_mean = np.mean(x,axis=0)
# y_mean = np.mean(y,axis = 0)
# print(x_mean,y_mean)
#What each class signifies in this model from n0-n9
dfAllClasses = pd.read_csv(PATH+'monkey_labels.txt')
label = [str.strip(x) for x in list(dfAllClasses[dfAllClasses.columns[2]])]
label
#Different visualization for train and test Data
valCnt = []
trainCnt = []

def countSamples(loader):
    temp = []
    for cls in range(10):
        tempCnt = 0
        for x in loader:
            if(cls == x[1]):
                tempCnt+=1

        temp.append(tempCnt)
    return temp    

trainCnt = countSamples(trainDataLoader.dataset.imgs)
valCnt = countSamples(valDataLoader.dataset.imgs)

#DATAFRAME
df = pd.DataFrame(data={'Class':label,'Train samples':trainCnt,'Val samples':valCnt})

print(df)
#Line bars
#Train set
fig0, ax0 = plt.subplots()
ax0.barh(label,trainCnt)
ax0.set_title('Training Data')
ax0.set_xlabel('Training Samples')
ax0.set_ylabel('Monkey categories')

#Validation set
fig1, ax1 = plt.subplots()
ax1.barh(label,valCnt)
ax1.set_title('Validation data')
ax1.set_xlabel('Validation samples')
ax1.set_ylabel('Monkey categories')
# ploting single image

#PLOTTING un-normalized data
temp = trainDataLoader.dataset[1][0].numpy()
temp = [0.1970754,0.1970754,0.1970754]*np.transpose(temp,axes=[1,2,0])+ [0.39099613,0.39099613,0.39099613]
plt.imshow(temp)
#PLOTTING normalized data
temp = trainDataLoader.dataset[1][0].numpy()
# temp = [0.1970754,0.1970754,0.1970754]*np.transpose(temp,axes=[1,2,0])+ [0.39099613,0.39099613,0.39099613]
plt.imshow(np.transpose(temp,axes=[1,2,0]))
#Creating GPU device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
#Creating train function
# forward => loss => backward => update weights

def train_model(model,mode,decay,criterion,optimizer,scheduler,dataloader,dictionary,num_epochs=12):
    correct = 0
    total = 0
    LR = 0
    totalLoss = []
    totalPrediction = []
    totalLRs = []
    for epoch in range(num_epochs):
        print("Epoch {}/{} ".format(epoch,num_epochs-1),flush=False)
        scheduler.step()
        
        if(decay == True):
            for params in optimizer.param_groups:
                LR = params['lr'] * (0.1**(epoch//7))
                params['lr'] = LR
            totalLRs.append(LR)
            
        if(mode == True):
            model.train()
        else:
            model.eval()
        
        for batch_id,(image,label) in enumerate(dataloader):
            optimizer.zero_grad()
            
            image = image.to(device)
            label  = label.to(device)
            
            outputs = model(image)     #forward-propogation
            _, predictionIndex = torch.max(outputs,1)    #predicted index
            loss = criterion(outputs,label)
            print("Batch "+str(batch_id)+" Loss = {0:.5f}".format(loss),end='\r',flush=True)
            
            correct += (predictionIndex == label).sum().item()
            total += label.size(0)
            totalPrediction.append(correct)
            
            loss.backward()     # back-propogation
            optimizer.step()  #update weights
            
            del image, label
        
        totalLoss.append(loss)
        torch.cuda.empty_cache()
    
    dictionary['totalLoss'] = totalLoss
    dictionary['prediction'] = totalPrediction
    dictionary['correct'] = correct
    dictionary['totalSize'] = total
    dictionary['LRs'] = totalLRs
    
    return model, dictionary
model_ft = torchvision.models.vgg16(pretrained=True)
model_ft = model_ft.to(device)

model_ft.features.requires_grad = False
model_ft.classifier.requires_grad = True

#Changing the last layer of model
model_ft.classifier[6].out_features = 10

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model_ft.classifier.parameters(),lr=0.008)

# exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=7,eta_min=0)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=7,gamma=0.1)
#Also u can use torch.optim.lr_scheduler.CosineAnnealingLR
#plot loss vs lr graph.
#Training
dictModel = {}
model_ft, dictModel = train_model(model_ft,mode=True,decay=False,criterion=criterion,optimizer=optimizer,dictionary=dictModel,scheduler=exp_lr_scheduler,dataloader=trainDataLoader,num_epochs=4)
plt.plot(dictModel['totalLoss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over eopchs')
print("Accuracy = ",100*(dictModel['correct']/dictModel['totalSize']))
#Saving model_ft
torch.save(obj=model_ft.state_dict(),f='./prototype1_dict.pth')
torch.save(obj=model_ft,f='./prototype1.pth')
dictModelVal = {}
model_val, dictModelVal = train_model(model_ft,mode=False,decay=False,criterion=criterion,optimizer=optimizer,dictionary=dictModelVal,scheduler=exp_lr_scheduler,dataloader=valDataLoader,num_epochs=1)
print("Accuracy validation set = ",100*dictModelVal['correct']/dictModelVal['totalSize'])
