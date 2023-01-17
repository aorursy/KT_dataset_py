import torch
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
%matplotlib inline
athlete = pd.read_csv(filepath_or_buffer='../input/athlete_events.csv')
noc_regions = pd.read_csv(filepath_or_buffer='../input/noc_regions.csv')
athlete.head(5)
#Dropping records with NaN
dropped_nan_height = athlete["Height"].dropna()
dropped_nan_weight = athlete["Weight"].dropna()
dropped_nan_age = athlete["Age"].dropna()
#Figure
fig = pyplot.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
pyplot.subplots_adjust(left=1, right=3)
#Plots
sns.distplot(dropped_nan_height, ax=ax1)
sns.distplot(dropped_nan_weight, ax=ax2)
sns.distplot(dropped_nan_age, ax=ax3)
pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Height",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Height"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)
pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Weight",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Weight"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)
pyplot.subplot(2, 3, 1)

sns.boxplot(x="Sex", y="Age",data=athlete[athlete["Season"] == 'Summer']).set_title("Season = Summer")
pyplot.subplot(2, 3, 2)
pyplot.title = "Winter"
sns.boxplot(x="Sex", y="Age"
            , data=athlete[athlete['Season'] == 'Winter']).set_title("Season = Winter")
pyplot.subplots_adjust(left=1, right=3, bottom=1, top=3)
data_with_predict = ["Season", "Weight", "Height", "Sex","Age"]
features = ["SeasonBinary", "Weight", "Height", "Age"]
data = athlete[data_with_predict].dropna()
data["BinarySex"] = (lambda x:  [1 if t=="M" else 0 for t in x])(data["Sex"])
data["SeasonBinary"] = (lambda x:  [1 if t=="Summer" else 0 for t in x])(data["Season"])
from sklearn.model_selection import train_test_split
from sklearn import tree
train, test = train_test_split(data, test_size=0.2)

X_train = train.as_matrix(columns=features)
Y_train = train.as_matrix(columns=["BinarySex"]).flatten()
X_test = test.as_matrix(columns=features)
Y_test = test.as_matrix(columns=["BinarySex"]).flatten()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
clf.predict(X_test)
acc_percentage = np.array(Y_test == clf.predict(X_test)).astype(np.int).sum()/len(Y_test) * 100
print("Accuracy on Test: {:3.4f}%".format(acc_percentage))
train, test = train_test_split(data, test_size=0.2)
valid_idx = int(len(train) * 0.1)
valid = train[:valid_idx]

class OlympicsDataSet(Dataset):
    def __init__(self,train,transforms=None):
        self.X = train.as_matrix(columns=features)
        self.Y = train.as_matrix(columns=["BinarySex"]).flatten()
        self.count = len(self.X)
        # get iterator
        self.transforms = transforms

    def __getitem__(self, index):
        nextItem = Variable(torch.tensor(self.X[index]).type(torch.FloatTensor))

        if self.transforms is not None:
            nextItem = self.transforms(nextItem[0])

        # return tuple but with no label
        return (nextItem, self.Y[index])

    def __len__(self):
        return self.count # of how many examples(images?) you have
olympicDS = OlympicsDataSet(train)
validDS = OlympicsDataSet(valid)
train_loader = torch.utils.data.DataLoader(olympicDS,
            batch_size=250, shuffle=False)
valid_loader = torch.utils.data.DataLoader(validDS,
            batch_size=1, shuffle=False)
testDS = OlympicsDataSet(test)
test_loader = torch.utils.data.DataLoader(testDS,
            batch_size=1, shuffle=False)
epochs = 10
class DNN(nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, num_classes):
        super(DNN, self).__init__()
        self.z1 = nn.Linear(input_size, first_hidden_size) # wx + b
        self.relu = nn.ReLU()
        self.z2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.z3 = nn.Linear(second_hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.z1(x) # input
        out = self.relu(out)

        out = self.z2(out) # first hidden layer
        out = self.relu(out)

        out = self.z3(out) # second hidden layer

        out = self.log_softmax(out) # output
        return out

    def name(self):
        return "DNN"
def train_dnn(net, trainL, validL):
    count = 0
    accuList = []
    lossList = []
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for epc in range(1,epochs + 1):
        print("Epoch # {}".format(epc))
        vcount = 0
        total_loss = 0
        net.train()
        for data,target in trainL:
            optimizer.zero_grad()
            out = net(data)
            loss = F.nll_loss(out, target, size_average=False)
            pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            count += pred.eq(target.data.view_as(pred)).sum()
            # Backward and optimize
            loss.backward()
            # update parameters
            optimizer.step()
        net.eval()
        for data, target in validL:
            out = net(data)
            loss = F.nll_loss(out, target, size_average=False)
            total_loss += loss.item()
            pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            vcount += pred.eq(target.data.view_as(pred)).sum().item()
        
        accuList.append(100. * (vcount / len(validL)))
        lossList.append(total_loss / len(validL))
    
    return accuList, lossList
     
myNet = DNN(4, 8, 4, 2)
accuList, lossList = train_dnn(myNet, train_loader, valid_loader)
def test(net, loader):
    net.eval()
    vcount = 0
    count = 0
    total_loss = 0.0
    for data, target in loader:
        out = net(data)
        loss = F.nll_loss(out, target, size_average=False)
        total_loss += loss.item()
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        vcount += pred.eq(target.data.view_as(pred)).sum().item()
    return 100. * (vcount / len(loader)), total_loss / len(loader)
test_acc, test_loss = test(myNet, test_loader)
print("The test set accuracy is {:3.4f}% \n The average loss is : {}".format(test_acc, test_loss))
pyplot.figure()
pyplot.plot(range(1, epochs + 1), accuList, "b--", marker="o", label='Validation Accuracy')
pyplot.legend()
pyplot.show()
pyplot.figure()
pyplot.plot(range(1, epochs + 1), lossList, "r", marker=".", label='Validation Loss')
pyplot.legend()
pyplot.show()