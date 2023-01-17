def numpy_ewma_vectorized_v2(data, window):



    alpha = 2 /(window + 1.0)

    alpha_rev = 1-alpha

    n = data.shape[0]



    pows = alpha_rev**(np.arange(n+1))



    scale_arr = 1/pows[:-1]

    offset = data[0]*pows[1:]

    pw0 = alpha*alpha_rev**(n-1)



    mult = data*pw0*scale_arr

    cumsums = mult.cumsum()

    out = offset + cumsums*scale_arr[::-1]

    return out
print(len(dataListM10[20000]))

data = dataListM10[1]

waveletname = 'db5'



plt.plot(data)

plt.show()



fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(18,18))

# for ii in range(3):

ii = 0

coeff_d = pywt.swt(data, waveletname, level = 3)



# print(len(coeff_d))

# for i in range(3):

#     print(len(coeff_d[i]))

#     print(coeff_d[i])



for ii in range(3):    

    axarr[ii, 0].plot(coeff_d[ii][0], 'r')

    axarr[ii, 1].plot(coeff_d[ii][1], 'g')



    #     if ii == 0:

    #         axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)

    #         axarr[ii, 1].set_title("Detail coefficients", fontsize=14)



plt.tight_layout()

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.
# from google.cloud import bigquery

# client = bigquery.Client()
print(os.listdir('../input'))

import scipy.io

mat1 = scipy.io.loadmat('../input/part_1.mat')

# print(mat1)
from scipy.signal import find_peaks

from scipy.signal import resample
def normalize(arr):

    return (arr- np.min(arr))/(np.max(arr) - np.min(arr))

testMat = scipy.io.loadmat('../input/part_{}.mat'.format(1))['p']

tempMat = testMat[0, 0]

mean = np.mean(tempMat[0, 0])

std = np.std(testMat[0, 0])

meanStd = mean/std

print(mean, " hahaha ", std, "lalala", meanStd)

tempSize = 125

dataTest = []

for i in range(1000):

    tempMat = testMat[0, i]

    tempLength = tempMat.shape[1]

    for j in range((int)(tempLength/tempSize)):

        tempPpg = tempMat[0, j*tempSize:(j+1)*tempSize]

        dataTest.append(tempPpg)

        

        
def test(aTest, meanStd):

    tempSum = 0

    mean = np.mean(aTest)

    std = np.std(aTest)

    for i in range(125):

        tempSum += np.power((aTest[i] - mean/std), 3)

    return tempSum/125
print(len(dataTest))

numAb = 0

for j in range(10):

    tempSum = 0

    aTest = dataTest[j*1200]

    res = test(aTest, meanStd)

    numAb += 1

    plt.plot(aTest)

    plt.show()

    print(res)



print(numAb)
import scipy.io

dataListM10 = []

target = []

size = 1000

for k in range(0):

    testMat = scipy.io.loadmat('../input/part_{}.mat'.format(k+1))['p']

    for i in range(1000):

        tempMat = testMat[0, i]

        tempLength = tempMat.shape[1]

        tempMat[0, :] = normalize(tempMat[0, :])

#         tempMat[2, :] = normalize(tempMat[2, :])

        for j in range((int)(tempLength/size)):

            tempPpg = tempMat[0, j*size:(j+1)*size]

#             tempPpg = tempMat[0, j*size:(j+1)*size].reshape(18, 18)



#             tempEcg = tempMat[2, j*size:(j+1)*size].reshape(18, 18)

            tempBp = tempMat[1, j*size:(j+1)*size]

#             dataListM10.append(np.r_[tempPpg, tempEcg].reshape(32, 32))

            dataListM10.append(tempPpg)

            target.append(tempBp)

        
tempSize = 125

dataTest = []

testMat = scipy.io.loadmat('../input/part_{}.mat'.format(1))['p']

for i in range(1000):

    tempMat = testMat[0, i]

    tempLength = tempMat.shape[1]

    for j in range((int)(tempLength/tempSize)):

        tempPpg = tempMat[0, j*tempSize:(j+1)*tempSize]

        dataTest.append(tempPpg)

        

        
print(len(dataTest))
plt.plot(dataTest[2])

plt.show()



tempSum = 0

aTest = dataTest[2]



mean = np.mean(testMat[0, 0])

std = np.std(testMat[0, 0])

meanStd = mean/std





print(mean, " hahaha ", std, "lalala", meanStd)

for j in range(25):

    tempSum = 0

    aTest = dataTest[j]

    for i in range(125):

        tempSum += np.power((aTest[i] - meanStd), 3)

    res = tempSum/125

    plt.plot(aTest)

    plt.show()

    print(res)
tempSize = 125

dataTest0 = []

testMat0 = scipy.io.loadmat('../input/part_{}.mat'.format(12))['p']

for i in range(0):

    tempMat = testMat0[0, i]

    tempLength = tempMat.shape[1]

    for j in range((int)(tempLength/tempSize)):

        tempPpg = tempMat[0, j*tempSize:(j+1)*tempSize]

        dataTest0.append(tempPpg)

print(len(dataTest0))
# for i in range(1000):

#     print(testMat0[0,i].shape)

# for i in range(1000):

#     plt.plot(testMat0[0, i][0, 0:125])

#     plt.show()
mean = np.mean(testMat[0, 0])

std = np.std(testMat[0, 0])

meanStd = mean/std

print(testMat0[0,0].shape)

print(len(dataTest0))

numAb = 0

for j in range(0):

    tempSum = 0

    aTest = dataTest0[j*1200]

    if test(aTest, meanStd) == False:

#     for i in range(125):

#         tempSum += np.power((aTest[i] - meanStd), 3)

#     res = tempSum/125

#     if (res > 4 or res < 2):

        numAb += 1

        plt.plot(aTest)

        plt.show()

        print(res)



print(numAb)
print(len(dataListM10))

print(len(target))

# plt.plot(target[500000])

# plt.show()
import os

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as td

import torchvision as tv

from torch.autograd import Variable



from torch.utils.data import DataLoader

import torch.nn.init as torch_init



from PIL import Image

# import nntools as nt

if torch.cuda.is_available:

    device = torch.device("cuda")

print(device)
class dataSetABP(td.Dataset):

    def __init__(self, dataList, target, M=32, mode = 'train'):

        super(td.Dataset, self).__init__()

        self.mode = mode

        self.dataList = dataList

        self.target = target

        self.M = M

        self.trainLength = int(0.8*len(self.dataList))

        self.valLength = int(0.1*len(self.dataList))

    def __len__(self):

        if self.mode == 'train':

            return int(0.8*len(self.dataList))

        else:

            return int(0.1*len(self.dataList))

    def __getitem__(self, idx):



        if self.mode == 'train': # np.expand_dims(x, axis=1)

            return torch.from_numpy(np.expand_dims(self.dataList[idx], axis=0)).type(torch.FloatTensor), torch.from_numpy(self.target[idx]).type(torch.FloatTensor)

#             idxStart = idx*self.M # 

#             return torch.from_numpy(np.stack(self.dataList[idxStart:idxStart+self.M])).type(torch.FloatTensor), torch.from_numpy(np.stack(self.target[idxStart:idxStart+self.M])).type(torch.FloatTensor)

        elif self.mode == 'val':

            return torch.from_numpy(np.expand_dims(self.dataList[idx+self.trainLength], axis=0)).type(torch.FloatTensor), torch.from_numpy(self.target[idx+self.trainLength]).type(torch.FloatTensor)

#             idxStart = (idx+self.trainLength)*self.M

#             return torch.from_numpy(np.stack(self.dataList[idxStart:idxStart+self.M])).type(torch.FloatTensor), torch.from_numpy(np.stack(self.target[idxStart:idxStart+self.M])).type(torch.FloatTensor)

        else:

            return torch.from_numpy(np.expand_dims(self.dataList[idx+self.trainLength+self.valLength], axis=0)).type(torch.FloatTensor), torch.from_numpy(self.target[idx+self.trainLength+self.valLength]).type(torch.FloatTensor)

#             idxStart = (idx+self.trainLength+self.valLength)*self.M

#             return torch.from_numpy(np.stack(self.dataList[idxStart:idxStart+self.M])).type(torch.FloatTensor), torch.from_numpy(np.stack(self.target[idxStart:idxStart+self.M])).type(torch.FloatTensor)



            
# trainSet = dataSetABP(dataListM10, target, M=32, mode = 'train')

# valSet = dataSetABP(dataListM10, target, M=32, mode = 'val')

# testSet = dataSetABP(dataListM10, target, M=32, mode = 'test')

trainLoader = DataLoader(dataSetABP(dataListM10, target, M=32, mode = 'train'), batch_size = 128, shuffle = True, pin_memory = True, drop_last = True)

valLoader = DataLoader(dataSetABP(dataListM10, target, M=32, mode = 'val'), batch_size = 128, shuffle = False, pin_memory = True, drop_last = True)

testLoader = DataLoader(dataSetABP(dataListM10, target, M=32, mode = 'test'), batch_size = 128, shuffle = True, pin_memory = True, drop_last = True)



# for ppgEcg, target0 in enumerate(trainLoader):

    

#     print(ppgEcg.shape)

class Resnet18Transfer(nn.Module):

    def __init__(self, num_classes, fine_tuning=True):

        super(Resnet18Transfer, self).__init__() 

        resnet = tv.models.resnet18(pretrained=False) 

        for param in resnet.parameters():

            param.requires_grad = fine_tuning

            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size = 7)

            self.bn1 = resnet.bn1

            self.relu = resnet.relu

            self.maxpool = resnet.maxpool

            self.layer1 = resnet.layer1

            self.layer2 = resnet.layer2

            self.layer3 = resnet.layer3

            self.layer4 = resnet.layer4

            self.avgpool = resnet.avgpool

            num_ftrs = resnet.fc.in_features

            self.fc = nn.Linear(num_ftrs, num_classes) 

            

    def forward(self, x):

        h = self.conv1(x)

        h = self.bn1(h)

        h = self.relu(h)

        h = self.maxpool(h)

        h = self.layer1(h)

        h = self.layer2(h)

        h = self.layer3(h)

        h = self.layer4(h)

        h = self.avgpool(h) 

        h = h.view(-1, 512) 

        y = self.fc(h) 

        return y
model = Resnet18Transfer(size).to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# model = annLstm(M = 32).to(device)

# criterion = torch.nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
def validation(model, valLoader):

    with torch.no_grad():

        mean_loss = 0

        for j, [ppgEcg, target0] in enumerate(valLoader):

            ppgEcg, target0 = ppgEcg.to(device), target0.to(device)

            outputs = model(ppgEcg)

            outputs = torch.squeeze(outputs, dim = 0)

            loss = criterion(outputs, target0)

            mean_loss += loss.item()

        return mean_loss / len(valLoader)

            
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

timeIter = 0

tempLoss = 0

# hidden = model.init_hidden()

for i in range(timeIter):

    print("i", i)

    if i > 0:

        for g in optimizer.param_groups:

            g['lr'] = 0.0001

    for j, [ppgEcg, target0] in enumerate(trainLoader):

#         print("hello j = ", j)

        ppgEcg, target0 = ppgEcg.to(device), target0.to(device)

        optimizer.zero_grad()

        outputs = model(ppgEcg)

        outputs = torch.squeeze(outputs, dim = 0)

#         target0 = target0.view(-1, 256)

#         print(outputs.size())

#         print(target0.detach().size())

#         print("target ", target[0])

#         print("outputs", outputs[0])

        loss = criterion(outputs, target0)

        loss.backward()

        optimizer.step()

        tempLoss += loss.item()

        if j % 500 == 0 and j != 0:

            print("j", j, tempLoss/500)

            valLoss = validation(model, valLoader)

            print("val ", valLoss)

#             print("target ", target0[:5])

#             print("outputs", outputs[:5])

#             print(model.fc1.weight[:5, :5])

            tempLoss = 0.0
for j, [ppgEcg, target0] in enumerate(testLoader):

        ppgEcg, target0 = ppgEcg.to(device), target0.to(device)

        print(ppgEcg.size())

        optimizer.zero_grad()

        outputs = model(ppgEcg)

#         outputs = torch.squeeze(outputs, dim = 0)

        if j == 3:

            break;

            

testNum = size



ground_truth = target0.detach().cpu().numpy()

prediction = outputs.detach().cpu().numpy()



print(ground_truth.shape)



testCase = 10

for i in range(12):

    plt.figure(figsize=(20,10))

    plt.plot(ground_truth[i+20, :testNum], label = "ground")

    plt.plot(prediction[i+20, :testNum], label = "prediction")

    plt.legend(fontsize=20)

    plt.show()