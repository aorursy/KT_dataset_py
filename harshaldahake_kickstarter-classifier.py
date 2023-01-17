import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.fc1 = nn.Linear(5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return torch.sigmoid(x)
        #return F.log_softmax(x, dim=1)
train = pd.read_csv("2018_kickstarter_train_data.csv")
newValues = np.delete(train.values, 6, axis=1) 

completeDataset = torch.tensor(newValues)


completeDatasetCat = completeDataset[completeDataset[:, 3] == 14]


completeDatasetSplit = torch.split(completeDataset, [5, 1], dim=1)

datasetX = completeDatasetSplit[0]
datasetY = completeDatasetSplit[1]

datasetAverageX = torch.mean(datasetX, 0, True)
datasetMaxX = torch.max(datasetX, 0, True)
datasetMinX = torch.min(datasetX, 0, True)

#datasetX = (datasetX - datasetAverageX[0]) / (datasetMaxX[0][0] - datasetMinX[0][0])


trainDataset = torch.utils.data.TensorDataset(datasetX, datasetY)

trainSet, testSet = torch.utils.data.dataset.random_split(trainDataset, [int(len(trainDataset) * (8/10)), 
                                                                         len(trainDataset) 
                                                                         - (int(len(trainDataset) * (8/10)))])
print(len(trainDataset))
print(len(trainSet))
print(len(testSet))

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = len(trainSet), shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = len(testSet))
globalTestLoader = torch.utils.data.DataLoader(trainDataset, batch_size = len(trainDataset))
def trainModel(net, trainLoader):
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    EPOCHS = 3

    for epoch in range(EPOCHS):
        for x, y in trainLoader:
            net.zero_grad()
            output = net(x.float())
            loss = F.binary_cross_entropy(output, torch.squeeze(y.float()))
            loss.backward()
            optimizer.step()
            print(loss)
    
def checkAccuracy(net, testLoader):
    correct = 0
    total = 0

    with torch.no_grad():
        x,y = iter(testLoader).next()
        output = net(x.float())
        output = (output > 0.5).float()
        for index, i in enumerate(output):
            if i == y[index]:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))
    return round(correct/total, 3)
print((len(trainDataset) // 4) * 4)
len(trainDataset)
def federatedLearning():
    netArray = [Net() for x in range(15)]
    dataArrayTrain = []
    dataArrayTest = []
    
    for i in range(15):
        print(i)
        catData = completeDataset[completeDataset[:, 3] == i]
        
        dataSplit = torch.split(catData, [5, 1], dim=1)
        
        x = dataSplit[0]
        y = dataSplit[1]
        
        tensorData = torch.utils.data.TensorDataset(x, y)
        print(int(len(trainDataset) * (8/10)))
        trainSet, testSet = torch.utils.data.dataset.random_split(tensorData, [int(len(tensorData) * (8/10)), 
                                                                         len(tensorData) 
                                                                         - (int(len(tensorData) * (8/10)))])
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = len(trainSet), shuffle=True)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size = len(testSet))
        
        dataArrayTrain.append(trainLoader)
        dataArrayTest.append(testLoader)
        
    
        
        
    rounds = 10
    
    for j in range(rounds):
        for i, net in enumerate(netArray):
            trainModel(net, dataArrayTrain[i])
            print("Accuracy for net with category " + str(i) + ":" + str(checkAccuracy(net, dataArrayTest[i])))
        
        netAverage = netArray[0].state_dict()
        for i in netAverage.keys():
            for j in range(1, len(netArray)):
                netAverage[i] += (netArray[j].state_dict())[i] 
            netAverage[i] = torch.div(netAverage[i], len(netArray))
        
        testFinalNet = Net()
        testFinalNet.load_state_dict(netAverage)
        print("Accuracy of averaged net: " + str(checkAccuracy(testFinalNet, globalTestLoader)))
        
        netArray = []
        for i in range(15):
            finalNet = Net()
            finalNet.load_state_dict(netAverage)
            netArray.append(finalNet)
federatedLearning()
