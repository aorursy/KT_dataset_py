# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",sep=',',header=0)
train=train.replace({'Sex':{'male':0,'female':1}})

test = pd.read_csv("../input/test.csv",sep=',',header=0)
test=test.replace({'Sex':{'male':0,'female':1}})
age_value = train.Age.value_counts(normalize=True,dropna=True)
age_fill_t = np.random.choice(train.Age.dropna().values,size=len(train[train['Age'].isnull()]),replace=False)
age_fill2_t = pd.Series(data=age_fill_t,index=train[train.Age.isnull()].index)
train['Age'].fillna(age_fill2_t,inplace=True)
train.isnull().sum()
# deal with test data for submission
age_value = train.Age.value_counts(normalize=True,dropna=True)
age_fill_t = np.random.choice(train.Age.dropna().values,size=len(test[test['Age'].isnull()]),replace=False)
age_fill2_t = pd.Series(data=age_fill_t,index=test[test.Age.isnull()].index)
test['Age'].fillna(age_fill2_t,inplace=True)
test.Fare.fillna(train.Fare.mode()[0],inplace=True)
test.isnull().sum()
train['Cabin'].loc[~train.Cabin.isnull()] = 1
train['Cabin'].loc[train.Cabin.isnull()]=0
#train['Cabin'][~train.Cabin.isnull()]=1
#train['Cabin'][train.Cabin.isnull()]=0
#train['Cabin']=train['Cabin'].astype(int)
train = train.drop(['PassengerId','Name','Ticket'],1)
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map({'S':1,'Q':2,'C':3}).astype(int)
train['family']= train['SibSp'] + train['Parch'] + 1
train = train.drop(['SibSp','Parch'],1)
train['Age_bin']=pd.cut(train['Age'],[0,14,31,45,60,90],right=False)
train['Fare_bin'] = pd.qcut(train['Fare'], 4)
train.Fare_bin.value_counts().index
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train['Age_num']=label.fit_transform(train['Age_bin'])
train['Fare_num']=label.fit_transform(train['Fare_bin'])
train[:5]
train = train.drop(['Age_bin','Fare_bin','Age','Fare'],1)
from sklearn.model_selection import train_test_split
dtrain,dtest = train_test_split(train,test_size=0.2)
#dtrain.shape; dtest.shape
dtest,ans = dtest.drop(['Survived'],1),np.array(dtest['Survived'])
dtrain[:5]
X = dtrain.drop('Survived',1)
y = np.array(dtrain['Survived'])
from sklearn.neural_network import MLPClassifier
MLP_mdl = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
MLP_pred = MLP_mdl.fit(X,y).predict(dtest)
MLP_prec = np.sum(ans==MLP_pred)/len(dtest)
print('Precision from Neural Network model: ' + str(MLP_prec*100) + '%')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
NB_pred = gnb.fit(X, y).predict(dtest)
print("Number of mislabeled points out of a total %d points : %d" % (len(NB_pred),(ans != NB_pred).sum()))
NB_prec = np.sum(ans==NB_pred)/len(dtest)
print('Precision from Gaussian Naive Bayes model: ' + str(round(NB_prec*100,2)) + '%')
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(random_state=0,n_estimators=100)
RF.fit(X,y)
print("Feature importance:")
print(list(zip(train.columns, RF.feature_importances_)))
RF_pred = RF.predict(dtest)
RF_prec = np.sum(ans==np.round(RF_pred).astype(int))/len(dtest)
print('Precision from Ramdom Forest Regression model: ' + str(round(RF_prec*100,2)) + '%')
# Decision Tree model
from sklearn import tree
tree_mdl = tree.DecisionTreeRegressor()
tree_mdl.fit(X,y)
# precision from decision tree model
tree_pred = tree_mdl.predict(dtest)
tree_prec = np.sum(ans==np.round(tree_pred).astype(int))/len(dtest)
print('Precision from decision tree model: ' + str(round(tree_prec*100,2)) + '%')
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X, y)
y_pred = model.predict(dtest)
predictions = [round(value) for value in y_pred]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ans, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X, y)
y_pred = model.predict(dtest)
predictions = [round(value) for value in y_pred]
test['Cabin'][~test.Cabin.isnull()]=1
test['Cabin'][test.Cabin.isnull()]=0
test['Cabin']=test['Cabin'].astype(int)
test = test.drop(['PassengerId','Name','Ticket'],1)
test['Embarked'] = test['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].map({'S':1,'Q':2,'C':3}).astype(int)
test['family']= test['SibSp'] + test['Parch'] + 1
test = test.drop(['SibSp','Parch'],1)
test['Age_bin']=pd.cut(test['Age'],[0,14,31,45,60,90],right=False)
test['Fare_bin']= pd.cut(test['Fare'],[-0.001,7.91,14.454,31.0,512.3292],right=True)
test.Fare_bin.value_counts()
test[test.Fare_bin.isnull()]
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
test['Age_num']=label.fit_transform(test['Age_bin'])
test['Fare_num']=label.fit_transform(test['Fare_bin'])
test = test.drop(['Age','Fare','Age_bin','Fare_bin'],1)
test.isnull().sum()
#train = train.drop('Fare_num',1)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train.drop('Survived',1), train[['Survived']])
y_pred = model.predict(test)
predictions = [round(value) for value in y_pred]
example = pd.read_csv("../input/gender_submission.csv",sep=',',header=0)
data_to_submit = pd.DataFrame({
    'PassengerId':example['PassengerId'],
    'Survived':pd.Series(predictions)
})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
#from sklearn.neural_network import MLPClassifier
MLP_mdl = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
MLP_pred = MLP_mdl.fit(train.drop('Survived',1), train[['Survived']]).predict(test)
data_to_submit = pd.DataFrame({
    'PassengerId':example['PassengerId'],
    'Survived':pd.Series(MLP_pred)
})
data_to_submit.to_csv('MLP_result_20180901.csv', index = False)
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(random_state=0,n_estimators=100)
RF_pred = RF.fit(train.drop('Survived',1), train[['Survived']]).predict(test)
RF_pred = np.round(RF_pred).astype(int)
data_to_submit = pd.DataFrame({
    'PassengerId':example['PassengerId'],
    'Survived':pd.Series(RF_pred)
})
data_to_submit.to_csv('RF_result_20180901.csv', index = False)