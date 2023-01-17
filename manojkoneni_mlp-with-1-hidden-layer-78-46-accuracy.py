# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import pickle as pkl
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
# data preprocessing
train = pd.concat([train, pd.get_dummies(train['Pclass'], prefix = 'Pclass')], axis = 1)
train.drop(['Pclass'],axis =1, inplace = True )
train.drop(['PassengerId'], axis = 1, inplace = True)
train.drop(['Name'], axis = 1, inplace = True)
train['Age'].fillna(train['Age'].median(), inplace = True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
train['Fare'].fillna(train['Fare'].median(), inplace = True)
train.Age/=train.Age.max()
train.SibSp/= train.SibSp.max()
train.Parch/=train.Parch.max()
train.drop(['Ticket'], axis = 1, inplace = True)
train.Fare/=train.Fare.max()
#train = pd.concat([train, pd.get_dummies(train['Cabin'], prefix = 'Cabin', dummy_na = True)], axis = 1)
train.drop(['Cabin'], axis =1, inplace = True)
train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix = 'Embarked')], axis = 1)
train.drop(['Embarked'], axis =1, inplace = True)
train = pd.concat([train, pd.get_dummies(train['Sex'], prefix = 'Sex')], axis = 1)
train.drop(['Sex'], axis =1, inplace = True)
train.head()
#dataset prep..

cols = list(train.columns)
x_train_cols = cols[-12:]
x_train = train[x_train_cols]
x_train.head()
y_train = train['Survived']
#x_train = x_train.fillna(0)
x_valid = x_train[700:891]
y_valid = y_train[700:891]
x_train = x_train[:700]
y_train = y_train[:700]
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train.values, y_train.values, x_valid.values, y_valid.values)
)

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size = 5, shuffle = True)
valid_dl = DataLoader(valid_ds, batch_size = 10, shuffle = True)
def fit(epochs, train_dl, valid_dl, classifier, optimizer):
    prec, accuracy, tr_err, valid_err, f1, recall, train_accuracy = [],[],[],[],[], [],[]
    for epoch in range(epochs):
        error, valid_error, error1 = 0, 0, 0
        ttp, tfp, ttn, tfn = 0, 0, 0, 0
        for x,y in train_dl:
            classifier.zero_grad()
            optimizer.zero_grad()
            y_pred = classifier(x.float())
            error = loss(y_pred, y)
            '''tc_m = metrics.confusion_matrix(y_pred[:,1].detach().numpy(), y.detach().numpy())
            ttp+=tc_m[0,0]
            tfn+=tc_m[0,1]
            ttn+=tc_m[1,1]
            tfp+=tc_m[1,0]'''
            error1+= error
            error.backward()
            optimizer.step()
        #train_accuracy.append((ttp+ttn)/(ttp+ttn+tfn+tfp))
        
        with torch.no_grad():
            tp, fp, tn, fn = 0, 0, 0, 0
            for x,y in valid_dl:
                y_pred = classifier(x.float())
                y_pred[y_pred>0.5]=1
                y_pred[y_pred<=0.5]=0
                c_m = metrics.confusion_matrix(y_pred[:,1], y)
                if(c_m.shape[0]!=1):
                    valid_error += loss(y_pred,y)
                    tp+=c_m[0,0]
                    fn+=c_m[0,1]
                    tn+=c_m[1,1]
                    fp+=c_m[1,0]
            print("epoch: ", epoch, " training_error: ", error1.item()/160, " validation_error: ", valid_error.item()/19)
            tr_err.append(error1.item()/160)
            valid_err.append(valid_error.item()/19)
            prec.append(tp/(tp+fp))
            recall.append(tp/(tp+fn))
            f1.append(2*tp/(2*tp+fp+fn))
            accuracy.append((tp+tn)/(tp+fp+fn+tn))
            #print("precision: ",tp/(tp+fp), " recall: ",tp/(tp+fn), " f1-score: ", 2*tp/(2*tp+fp+fn), " accuracy: ", (tp+tn)/(tp+fp+fn+tn))
        if(epoch==epochs-1):
            torch.save(classifier, "./final.pkl")
    return (tr_err, valid_err, prec, recall, f1, accuracy, train_accuracy)
class classifier(nn.Module):
    def __init__(self, sigmoid, softmax, relu):
        super(classifier, self).__init__()
        self.layer1 = nn.Linear(12, 64)
        self.layer2 = nn.Linear(64, 2)
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.relu = relu
    
    def forward(self, x):
        x = self.layer1(x)
        #x = self.sigmoid(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x, dim=1)
        return x
        
#dev = torch.device(
#    "cuda") if torch.cuda.is_available() else torch.device("cpu")

classifier1 = classifier(torch.sigmoid, F.softmax, F.relu)
#classifier1.to(dev)
optimizer = torch.optim.Adagrad(params = classifier1.parameters(), lr = 3*1e-3)
loss = nn.CrossEntropyLoss()
#loss = nn.BCELoss()
tr_err, valid_err, prec, recall, f1, accuracy, train_accuracy = fit(200, train_dl, valid_dl, classifier1, optimizer)

_, (ax0,ax1) = plt.subplots(1, 2)
ax0.plot(np.arange(len(tr_err)) + 1, tr_err, 'b')
ax0.plot(np.arange(len(valid_err)) + 1, valid_err, 'r')
ax0.set(xlabel='Epochs', ylabel='error',
       title='Epoch vs error')
ax1.plot(np.arange(len(prec)) + 1, prec, 'g', label='train_accuracy')
ax1.plot(np.arange(len(prec)) + 1, f1, 'b', label = 'validation_accuracy')
#ax1.plot(np.arange(len(prec)) + 1, accuracy, 'r', label='accuracy')
ax1.set(xlabel='Epochs', ylabel='accuarcy',
       title='Epoch vs error')
ax0.grid()
ax1.grid()
plt.show() 
test = pd.concat([test, pd.get_dummies(test['Pclass'], prefix = 'Pclass')], axis = 1)
test.drop(['Pclass'],axis =1, inplace = True )
#test.drop(['PassengerId'], axis = 1, inplace = True)
test.drop(['Name'], axis = 1, inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)
test.Age/=test.Age.max()
test.SibSp/= test.SibSp.max()
test.Parch/=test.Parch.max()
test.drop(['Ticket'], axis = 1, inplace = True)
test.Fare/=test.Fare.max()
#train = pd.concat([train, pd.get_dummies(train['Cabin'], prefix = 'Cabin', dummy_na = True)], axis = 1)
test.drop(['Cabin'], axis =1, inplace = True)
test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix = 'Embarked')], axis = 1)
test.drop(['Embarked'], axis =1, inplace = True)
test = pd.concat([test, pd.get_dummies(test['Sex'], prefix = 'Sex')], axis = 1)
test.drop(['Sex'], axis =1, inplace = True)
test.head()
cols = list(test.columns)
ind_cols = cols[1:]
test_data = test[ind_cols]
labels = cols[0]
label = test[labels]
print(test_data.head(), label.head())
classifier = torch.load("./final.pkl")
predicted = classifier(torch.tensor(test_data.values).float())
final = [int(x) for x in predicted[:,1].detach().numpy()]
df = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':final})
#print(output)
print(df)
df.to_csv("submission_file.csv", index = False)