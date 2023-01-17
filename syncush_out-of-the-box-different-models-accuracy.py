import seaborn as sns
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
import os
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from collections import Counter
init_notebook_mode()
def print_acc(acc,model_name):
    print("{} validation accuracy is {:.4f}%".format(model_name, acc))

# Outlier detection 
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    # detect outliers from Age, SibSp , Parch and Fare
    # Drop outliers
    df = df.drop(multiple_outliers, axis = 0).reset_index(drop=True)
    return multiple_outliers

def prepare_data(path, is_test=False):
    data = pd.read_csv(path)
    data["Pclass"].astype('int32')
    if not is_test:
        data["Survived"].astype('int32')
    data["Parch"].astype('int32')
    data["SibSp"].astype('int32')
    data["Fare"].astype('float')
    data["Age"].astype('float')
    index_NaN_age = list(data["Age"][data["Age"].isnull()].index)
    for i in index_NaN_age :
        age_med = data["Age"].median()
        age_pred = data["Age"][((data['SibSp'] == data.iloc[i]["SibSp"]) & (data['Parch'] == data.iloc[i]["Parch"]) & (data['Pclass'] == data.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            data['Age'].iloc[i] = age_pred
        else :
            data['Age'].iloc[i] = age_med
    data["Embarked"].fillna('S')
    data["f_size"] = data["SibSp"] + data["Parch"] + 1
    data['Single'] = data['f_size'].map(lambda x: 1 if x == 1 else 0)
    data['SmallF'] = data['f_size'].map(lambda x: 1 if  x == 2  else 0)
    data['MedF'] = data['f_size'].map(lambda x: 1 if 3 <= x <= 4 else 0)
    data['LargeF'] = data['f_size'].map(lambda x: 1 if x >= 5 else 0)
    data['is_male'] = data['Sex'].map(lambda x: 1 if x == "male" else 0)
    data['is_female'] = data['Sex'].map(lambda x: 1 if x == "female" else 0)
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in data["Name"]]
    data["Title"] = pd.Series(dataset_title)
    data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    data["Embarked"] = data["Embarked"].fillna('S')
    data["Embarked"] = data["Embarked"].map({"S":1, "C":2, "Q":3})
    data["Embarked"] = data["Embarked"].astype(int)
    data["Title"] = data["Title"].astype(int)
    data.drop(columns=["Name", "Cabin", "Ticket"], inplace=True)
    if not is_test:
        detect_outliers(data, 2, ["Age","SibSp","Parch","Fare"])
    return data
data = prepare_data('../input/train.csv')
data.head(5)
features1 = ["Age","SibSp","Parch", "Pclass", "is_male", "is_female", "f_size", "Single", "SmallF", "MedF", "LargeF", "Title", "Embarked"]
target = ["Survived"]
print("Total number of features is {}".format(len(features1)))
test_csv = prepare_data('../input/test.csv', True)
test_data = test_csv[features1]
test_csv.head(5)
train, valid = train_test_split(data, test_size=0.2)
print("The Train Set Size is {} \nThe Validation Set Size is {}".format(len(train), len(valid)))
print("Test Set Size is {}".format(len(test_data)))
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

train_x , train_y = data[features1].as_matrix(), data[target].as_matrix()

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)

print((np.array(clf.predict(valid[features1].as_matrix()) == valid[target].as_matrix().flatten(), dtype=np.int).sum() * 100.) / len(valid))

result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': clf.predict(test_data.as_matrix())})
result.to_csv(path_or_buf='decision_tree_submittion.csv', index = False, header = True)
class TitanicLoader(Dataset):
    def __init__(self,train,transforms=None):
        self.X = train.as_matrix(columns=features1)
        self.Y = train.as_matrix(columns=target).flatten()
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
        return self.count
class DNN(nn.Module):
    def __init__(self, input_size, first_hidden_size, second_hidden_size, num_classes):
        super(DNN, self).__init__()
        self.z1 = nn.Linear(input_size, first_hidden_size)
        self.relu = nn.ReLU()
        self.z2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.z3 = nn.Linear(second_hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(first_hidden_size)
        self.bn2 = nn.BatchNorm1d(second_hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.z1(x) # input
        out = self.relu(out)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.z2(out) # first hidden layer
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.bn2(out)
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
epochs = 12
titanic_train_DS = TitanicLoader(train)
titanic_valid_DS = TitanicLoader(valid)

train_loader = torch.utils.data.DataLoader(titanic_train_DS,
            batch_size=6, shuffle=False)
valid_loader = torch.utils.data.DataLoader(titanic_valid_DS,
            batch_size=1, shuffle=False)
myNet = DNN(len(features1), 23, 4, 2)
accuList, lossList = train_dnn(myNet, train_loader, valid_loader)
pyplot.figure()
pyplot.plot(range(1, epochs + 1), accuList, "b--", marker="o", label='Validation Accuracy')
pyplot.legend()
pyplot.show()
pyplot.figure()
pyplot.plot(range(1, epochs + 1), lossList, "r", marker=".", label='Validation Loss')
pyplot.legend()
pyplot.show()
def get_preds(test, net):
    net.eval()
    preds = []
    for data, target in test:
        out = net(data)
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        preds.append(pred.item())
    return preds
test_data['Survived'] = -1
titanic_test_DS = TitanicLoader(test_data)
test_loader = torch.utils.data.DataLoader(titanic_test_DS,
            batch_size=1, shuffle=False)
result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': get_preds(test_loader, myNet)})
result.to_csv(path_or_buf='neural_network_submittion.csv', index = False, header = True)
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)
neigh.fit(train[features1].as_matrix(), train[target].as_matrix().flatten())
print_acc((np.array(neigh.predict(valid[features1].as_matrix()) == valid[target].as_matrix().flatten(), dtype=np.int).sum() * 100.) / len(valid), "K-NN")
result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': neigh.predict(test_data[features1].as_matrix())})
result.to_csv(path_or_buf='knn_submittion.csv', index = False, header = True)
gnb = GaussianNB()
y_pred = gnb.fit(train[features1].as_matrix(), train[target].as_matrix().flatten()).predict(valid[features1].as_matrix())
print_acc(float(np.array(y_pred == valid[target].as_matrix().flatten(), dtype=np.int).sum() * 100) / len(valid), "Naive Bayes")
result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': gnb.predict(test_data[features1].as_matrix())})
result.to_csv(path_or_buf='naive_bayes_submittion.csv', index = False, header = True)
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=25,random_state=1)
clf3 = GaussianNB(var_smoothing=True)
clf4 = LinearSVC(random_state=5)
gbm =  xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('xgb', gbm)], voting='soft')
eclf1 = eclf1.fit(train[features1].as_matrix(), train[target].as_matrix().flatten())
print_acc(float(np.array(eclf1.predict(valid[features1].as_matrix()) == valid[target].as_matrix().flatten(), dtype=np.int).sum() * 100) / len(valid), "Voting")
result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': eclf1.predict(test_data[features1].as_matrix())})
result.to_csv(path_or_buf='soft_voting_submittion.csv', index = False, header = True)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=600, learning_rate=0.05)
y_pred = gbm.fit(train[features1].as_matrix(), train[target].as_matrix().flatten()).predict(valid[features1].as_matrix())
print_acc(float(np.array(y_pred == valid[target].as_matrix().flatten(), dtype=np.int).sum() * 100) / len(valid), "XGB")
result = pd.DataFrame(data={'PassengerId': test_csv['PassengerId'], 'Survived': gbm.predict(test_data[features1].as_matrix())})
result.to_csv(path_or_buf='gbm_submittion.csv', index = False, header = True)