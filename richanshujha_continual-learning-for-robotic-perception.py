import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from PIL import Image
##LOADING THE MNIST DATASET
trainData = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
testData = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

mnistData = pd.concat([trainData, testData], axis=0)
print('Merged data loaded with shape: ', str(mnistData.shape))
labelData = mnistData['label'].to_numpy()
featureData = mnistData.drop(['label'], axis = 1)
featureData = featureData/255
featureData = featureData.to_numpy().reshape(featureData.shape[0], 28, 28)
print('Reshaped MNIST features shape: ', str(featureData.shape))
print('Reshaped MNIST labels shape: ', str(labelData.shape))
#SPLITS DATA INTO TRAINING AND TESTING PAIRS
def splitData(features, labels, ttRatio):
    n = int(features.shape[0] * ttRatio)
    xTrain = features[:n] 
    xTest = features[n:]
    yTrain = labels[0:n]
    yTest = labels[n:]
    print('Split the data')
    print('Training features shape: ', str(xTrain.shape))
    print('Training labels shape: ', str(xTest.shape))
    print('Testing features shape: ', str(yTrain.shape))
    print('Testing features shape: ', str(yTest.shape))
    return(xTrain, xTest, yTrain, yTest)

#ROTATION ACCORIDING TO THE CODE PROVIDED FOR THE PROJECT
def rotateData(d, rotation):
    result = torch.FloatTensor(d.shape[0], 784)
    tensor = transforms.ToTensor()

    for i in range(d.shape[0]):
        img = Image.fromarray(d[i])
        result[i] = tensor(img.rotate(rotation)).view(784)
    return result

#GENERATING TASKS. USING ROTATION OF THE DATASET FROM THE CODE PROVIDED IN THE MNIST DATASET LINK FOR THE PROJECT.
def createTasks(xTrain, xTest, yTrain, yTest, nTasks, minRot, maxRot):
    tasksTrain = []
    tasksTest = []
    for t in range(nTasks):
        minR = 1.0 * t / nTasks * (maxRot - minRot) + minRot
        maxR = 1.0 * (t + 1) / nTasks * (maxRot - minRot) + minRot
        rot = random.random() * (maxR - minR) + minR
        print('Creating Task ', str(t+1), ' With rotation %0.3f'%(rot))
        tasksTrain.append([rot, rotateData(xTrain, rot), yTrain])
        tasksTest.append([rot, rotateData(xTest, rot), yTest])
        
    print('Completed Creation of Tasks')
    return(np.array(tasksTrain), np.array(tasksTest))
xTrain, xTest, yTrain, yTest = splitData(featureData, labelData, 0.9)
##HELPER FUNCTION TO DISPLAY THE IMAGES TO CHECK OUT OUR DATA
def displayImages(features,labels, m, n):
    f, axarr = plt.subplots(m,n)
    f.set_figheight(3*m)
    f.set_figwidth(3*n)
    x = 0
    for i in range(m):
        for j in range(n):
            if(m == 1):
                axarr[j].imshow(features[x])
                axarr[j].set_title(labels[x])
                axarr[j].axis('off')
            elif(n == 1):
                axarr[i].imshow(features[x])
                axarr[i].set_title(labels[x])
                axarr[i].axis('off')
            else:
                axarr[i, j].imshow(features[x])
                axarr[i, j].set_title(labels[x])
                axarr[i, j].axis('off')
            x += 1
            
displayImages(xTrain, yTrain, 2,5)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def train(model, device, x_train, t_train, optimizer, epoch):
    model.train()
    
    for start in range(0, len(t_train)-1, 256):
      end = start + 256
      x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      output = model(x)
      loss = F.cross_entropy(output, y)
      loss.backward()
      optimizer.step()
        
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

def test(model, device, x_test, t_test):
    model.eval()
    test_loss = 0
    correct = 0
    for start in range(0, len(t_test)-1, 256):
      end = start + 256
      with torch.no_grad():
        x, y = torch.from_numpy(x_test[start:end]), torch.from_numpy(t_test[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

    print('Testing Accuracy: ' + str(correct) + '/' + str(len(t_test)) + ' -> %0.3f'%(correct / len(t_test)))
    return (100 * correct / len(t_test))

tasksTrain, tasksTest = createTasks(xTrain, xTest, yTrain, yTest, 3, 0, 90)
print(tasksTrain.shape)
# CONVERTS THE TASKS WHICH HAVE TENSORS INTO PURELY ND ARRAY.
def npTasks(tasks):
    res = []
    for t in range(len(tasks)):
        res.append(np.array(tasks[t][1]).reshape(tasks[t][1].shape[0],28,28))
    return(np.array(res))

npTrain = npTasks(tasksTrain)
npTest = npTasks(tasksTest)

#SHOWING ROTATION TASKS (5 Per Task)
displayImages(npTrain[0], yTrain, 1, 5)
displayImages(npTrain[1], yTrain, 1, 5)
displayImages(npTrain[2], yTrain, 1, 5)
## WITHOUT CONTINUAL LEARNING
use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
model.float()
torch.manual_seed(1)

Results = []

#The model will be trained for each task and tested against the combined test sets of all tasks

def noCLTrainTest(xTrain, yTrain, xTest, yTest, epochs):
    res = []
    model = Net().to(device)
    model.float()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        train(model, device, xTrain.reshape(xTrain.shape[0],1,28,28) , yTrain, optimizer, epoch)
        print('')
        res.append(test(model, device, xTest.reshape(xTest.shape[0],1,28,28) , yTest))
    return(res)

print('--------')
#TASK 1
xTr, yTr = npTrain[0], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(noCLTrainTest(xTr ,yTr , xTe, yTe, 3))

print('--------')
#TASK 2
xTr, yTr = npTrain[1], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(noCLTrainTest(xTr ,yTr , xTe, yTe, 3))

print('--------')
#TASK 3
xTr, yTr = npTrain[2], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(noCLTrainTest(xTr ,yTr , xTe, yTe, 3))
## USING ELASTIC WEIGHTS CONSOLIDATION (REGULARIZATION)
fisherDict = {}
optparDict = {}
ewcLambda = 0.8

## CALLED FOR EACH TASK TO UPDATE THE FISHER AND OPTPAR DICTS
## THESE WILL BE USED TO CALCULATE THE LOSS
def updateGradientDicts(t, xTr, yTr):
    model.train()
    optimizer.zero_grad()

    #FINDING THE GRADIENTS
    for start in range(0, len(yTr)-1, 256):
        end = start + 256
        x, y = torch.from_numpy(xTr[start:end]), torch.from_numpy(yTr[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

    fisherDict[t] = {}
    optparDict[t] = {}

    #UPDATING THE GRADIENTS STORED IN FISHER AND OPTPAR DICT
    for name, param in model.named_parameters():
        optparDict[t][name] = param.data.clone()
        fisherDict[t][name] = param.grad.data.clone().pow(2)
    
#MODIFIED TRAINING METHOD WITH REGULARIZATION LOSS 
#CALCULATED USING THE FISHER AND OTPAR DICT VALUES)
def ewcTrain(model, device, t, xTr, yTr, optimizer, epoch):
    model.train()

    for start in range(0, len(yTr)-1, 256):
        end = start + 256
        x, y = torch.from_numpy(xTr[start:end]), torch.from_numpy(yTr[start:end]).long()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        
        ## ADDITION OF REGULARIZATION LOSS
        for task in range(t):
            for name, param in model.named_parameters():
                if task in fisherDict:
                    fisher = fisherDict[task][name]
                    optpar = optparDict[task][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewcLambda
        
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
## TRAINING AND TESTING THE CONTINUAL LEARNING MODEL USING EWC
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def ewcCLTrainTest(xTrain, yTrain, xTest, yTest, epochs, t):
    res = []
    for epoch in range(epochs):
        ewcTrain(model, device, t, xTrain.reshape(xTrain.shape[0],1,28,28) , yTrain, optimizer, epoch)
        res.append(test(model, device, xTest.reshape(xTest.shape[0],1,28,28) , yTest))
        
    updateGradientDicts(t, xTrain.reshape(xTrain.shape[0],1,28,28), yTrain)
    return(res)

print('--------')
#TASK 1
xTr, yTr = npTrain[0], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(ewcCLTrainTest(xTr ,yTr , xTe, yTe, 3, 0))

print('--------')
#TASK 2
xTr, yTr = npTrain[1], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(ewcCLTrainTest(xTr ,yTr , xTe, yTe, 3, 1))

print('--------')
#TASK 3
xTr, yTr = npTrain[2], yTrain
xTe = np.concatenate([npTest[0], npTest[1], npTest[2]])
yTe = np.concatenate([yTest, yTest, yTest])
Results.append(ewcCLTrainTest(xTr ,yTr , xTe, yTe, 3, 2))
## Plotting the Results
Results = np.array(Results).reshape(2,3,3)

#TO FORCE X TO TAKE INTEGER VALUES ONLY
X = [1,2,3]
from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
minVal, maxVal = min(Results.flatten()), max(Results.flatten())

plt.xlabel('Tasks')
plt.ylabel('Accuracy')
plt.ylim((minVal-minVal%10)-10, 10+(maxVal-maxVal%10))
plt.plot(X, Results[0,:,1], label = 'No CL')
plt.plot(X, Results[1,:,2], label = 'CL Using EWC')
plt.legend()
plt.show()