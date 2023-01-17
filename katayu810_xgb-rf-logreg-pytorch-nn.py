# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
submission = pd.read_csv("../input/titanic/gender_submission.csv")

submission.head()
train = pd.read_csv("../input/titanic/train.csv")

print(train.shape)

train.head()

train_len = len(train)
test = pd.read_csv("../input/titanic/test.csv")

print(test.shape)

test.head()

test_len = len(test)
import collections

data = pd.concat([train,test], sort = False)

Sex = collections.Counter(data["Sex"])

Embarked = collections.Counter(data["Embarked"])

Cabin = collections.Counter(data["Cabin"])

print(Embarked)

print(Cabin)



#前処理は欠損値の処理をしていく

#Cabin列は削除

#Embarkedの欠損値列はSで埋める
data.isnull().sum()
data["Sex"].replace(["male","female"], [0, 1], inplace = True)

data.head()
data["Fare"].fillna(np.mean(data["Fare"]), inplace = True)

data["Fare"].isnull().sum()
from sklearn.model_selection import train_test_split

data = data[["Survived","Pclass","Sex", "SibSp", "Parch", "Embarked","Fare"]]

data["Embarked"].fillna("S", inplace = True) #Sが多いので取り敢えずSで埋める

data["Embarked"].replace(["S","C", "Q"], [1,2,3], inplace=True)

data.isnull().sum()
X = data.iloc[:train_len]

y = X["Survived"]

X = X.drop("Survived", axis = 1)

test = data.iloc[train_len:]

test.drop("Survived", axis = 1, inplace = True)

test.head()
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth = 5, n_estimators = 5, random_state = 42)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_pred, y_test)

print("訓練時の精度"+str(score))

print("認識精度" +str(accuracy))
#再現率と適合率をみる

#f1-scoreは再現率と適合率の調和率

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#confusion matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')

plt.xlabel('predicted class')

plt.ylabel('true value')
#グリッドサーチ

parameters = {

    'n_estimators' :[3,5,10,30,50],

    'random_state' :[7,42],

    'max_depth' :[3,5,8,10],

}



from sklearn.model_selection import GridSearchCV

Grid = GridSearchCV(estimator= RandomForestClassifier(), param_grid=parameters, cv=3)

Grid.fit(X_train, y_train)
best_model = Grid.best_estimator_

train_score = best_model.score(X_train, y_train)

test_score = best_model.score(X_test, y_test)

print("訓練での認識精度:" + str(train_score))

print("テストデータでの認識精度   :" + str(test_score))



pred_y = best_model.predict(X_test)

accuracy = accuracy_score(pred_y, y_test)

print("テストデータの認識精度ver2:" + str(accuracy))

print(Grid.best_estimator_)
mat = confusion_matrix(y_test, pred_y)

sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')

plt.xlabel('predicted class')

plt.ylabel('true value')

#訓練データの認識精度が少し下がっていたけどこっちでは上がってる！！！！
#さらにグリッドサーチでパラメータを最適化してみる！

parameters = {

    'n_estimators' :[9,9.5,10,10.5,11],

    'random_state' :[7,42],

    'max_depth' :[2,2.5,3,3.5,4],

}



from sklearn.model_selection import GridSearchCV

Grid_2 = GridSearchCV(estimator= RandomForestClassifier(), param_grid=parameters, cv=3)

Grid_2.fit(X_train, y_train)
best_model_2 = Grid_2.best_estimator_

train_score = best_model_2.score(X_train, y_train)

test_score = best_model_2.score(X_test, y_test)

print("訓練での認識精度:" + str(train_score))

print("テストデータでの認識精度   :" + str(test_score))



pred_y = best_model_2.predict(X_test)

accuracy = accuracy_score(pred_y, y_test)

print("テストデータの認識精度ver2:" + str(accuracy))

print(Grid.best_estimator_)
mat = confusion_matrix(y_test, pred_y)

sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')

plt.xlabel('predicted class')

plt.ylabel('true value')

#ちょっと認識精度下がった？？

#原因不明;;
#初めのGridのmodelを使って予測する



test.reset_index(drop = True, inplace =True)

test = test[["Pclass","Sex","SibSp","Parch","Embarked","Fare"]]
rlt_1_list = pd.read_csv("../input/titanic/gender_submission.csv")

rlt_1_list["Survived"] = list(map(int, rlt))

rlt_1_list.head()
#データの整えかたpart2

#rlt_2_list = pd.read_csv("../input/titanic/gender_submission.csv")

#rlt_2_list = pd.DataFrame({"PassengerId":rlt_2_list["PassengerId"],

                          #"Survived":rlt})
submission = pd.read_csv("../input/titanic/gender_submission.csv")

submission.head()
#続けてアンサンブルしてスコアを上げてみる！

#線形回帰

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty="l2",C=1.0,random_state = 42)

lr.fit(X_train, y_train)

print("訓練認識精度" + str(lr.score(X_train,y_train)))

print("テスト認識精度" + str(lr.score(X_test, y_test)))

parameters = {

    'C' :[0.5,0.8,1,1.3,1.5],

    'random_state' :[7,42,1000],

}



from sklearn.model_selection import GridSearchCV

Grid_lr = GridSearchCV(estimator= LogisticRegression(), param_grid=parameters, cv=4)

Grid_lr.fit(X_train, y_train)
best_lr = Grid_lr.best_estimator_

print("訓練:"+ str(best_lr.score(X_train, y_train)))

print("テスト:"+ str(best_lr.score(X_test, y_test)))

#C=1で良かったんか

y_pred = best_lr.predict(X_test)

rlt_2 = best_lr.predict(test)


print(classification_report(y_test, y_pred))

mat = confusion_matrix(y_test, pred_y)

sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='d', cmap='RdPu')

plt.xlabel('predicted class')

plt.ylabel('true value')
rlt_2_list = pd.read_csv("../input/titanic/gender_submission.csv")

rlt_2_list["Survived"] = list(map(int, rlt_2))

rlt_2_list.head()
import xgboost as xgb

parameters = {

    'n_estimators' :[2,5,8,10,13],

    'random_state' :[7,42,100],

    'max_depth' :[2,4,6,8,10],

}



model_xgb = GridSearchCV(estimator= xgb.XGBClassifier(), param_grid=parameters, cv = 5)

model_xgb.fit(X_train, y_train)
best_xgb = model_xgb.best_estimator_

print(best_xgb.score(X_test, y_test))

rlt_3 = best_xgb.predict(test)
rlt_3_list = pd.read_csv("../input/titanic/gender_submission.csv")

rlt_3_list["Survived"] = list(map(int, rlt_3))

rlt_3_list.head()
esb = pd.read_csv("../input/titanic/gender_submission.csv")

esb["Survived"] = rlt_1_list["Survived"] + rlt_2_list["Survived"] + rlt_3_list["Survived"]

esb.head()
esb["Survived"] = (esb["Survived"] >= 2).astype(int)

#2以上はTrueなので1
esb.to_csv("submission1.csv", index = False)
from torch import nn, optim

import torch

from torch.utils.data import TensorDataset, DataLoader

from statistics import mean



X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 10)



net = nn.Sequential(

    nn.Linear(6,20),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(20,20),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(20,1)

)



loss_fn = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(net.parameters())

#損失関数のログ

train_losses = []

test_losses = []

correct_rate = []



X_train = torch.tensor(X_train.values, dtype = torch.float32)

y_train = torch.tensor(y_train.values, dtype = torch.float32)

X_test = torch.tensor(X_test.values, dtype = torch.float32)

y_test = torch.tensor(y_test.values, dtype = torch.float32)





#ミニバッチ用

ds = TensorDataset(X_train, y_train)

loader = DataLoader(ds, batch_size = 64, shuffle=True)
for epoc in range(100):

    running_loss = 0.0

    

    #trainモード

    net.train()

    for i,(xx, yy) in enumerate(loader):

        optimizer.zero_grad()

        y_pred = net(X_train)

        

        loss = loss_fn(y_pred.view_as(y_train),y_train)

        loss.backward()

        

        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / i)

    

    #評価モード

    net.eval()

    h = net(X_test)

    # シグモイド関数を作用させた結果はy=1の確率を表す

    prob = nn.functional.sigmoid(h)

    prob = prob.view(-1)

    test_loss = loss_fn(prob,y_test)

    test_losses.append(test_loss.item())

    # 確率が0.5以上のものをクラス1と予想し、それ以外を0とする

    # PyTorchにはBool型がないので対応する型としてByteTensorが出力される

    y_pred = prob > 0.6

    

    # 予測結果の確認 (yはFloatTensorなのでByteTensor

    # に変換してから比較する）

    correct_rate.append((y_test.byte() == y_pred.view_as(y_test)).sum().item()/ len(y_test))

print(mean(correct_rate))
plt.subplot()

plt.plot(train_losses)

plt.plot(test_losses, c="r")
class CustomLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = True, p = 0.4):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p)

        

    def forward(self, x):

        x = self.linear(x)

        x = self.relu(x)

        x = self.drop(x)

        return x



# Part1

#mlp = nn.Sequential(

    #CustomLinear(6,200),

    #CustomLinear(200,200),

    #CustomLinear(200,200),

    #nn.Linear(200, 1))
#Part2

class MyMLP(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.ln1 = CustomLinear(in_features, 200)

        self.ln2 = CustomLinear(200, 200)

        self.ln3 = CustomLinear(200, 200)

        self.ln4 = CustomLinear(200, out_features)

        

    def forward(self, x):

        x = self.ln1(x)

        x = self.ln2(x)

        x = self.ln3(x)

        x = self.ln4(x)

        return x



mlp = MyMLP(6,1)
mlp