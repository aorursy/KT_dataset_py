# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",sep=',',header=0)
test = pd.read_csv("../input/test.csv",sep=',',header=0)
dataset = pd.concat([train.drop('Survived',1),test],axis=0)
train[:5]
test[:5]
train.info()
sns.barplot(x='Pclass',y='Survived',data=train)
sns.barplot(x='Sex',y='Survived',data=train)
sns.barplot(x='Embarked',y='Survived',data=train)
sns.distplot(train.Age.dropna(),kde='True')
sns.distplot(test.Age.dropna(),kde='True')
train = pd.concat([train,pd.cut(train['Age'],bins=range(0,90,5)).rename('Age_lvl').to_frame()],axis=1)
sns.countplot(x='Age_lvl',data=train,hue='Survived')
sns.barplot(x='SibSp',y='Survived',data=train)
sns.barplot(x='Parch',y='Survived',data=train)
dataset.isnull().sum()
# categorical variables to numbers
dataset=dataset.replace({'Sex':{'male':0,'female':1}})
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)
dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
dataset['Embarked'] = dataset['Embarked'].map({'S':1,'Q':2,'C':3}).astype(int)
# new feature: has cabin or not 
dataset['Cabin'].loc[~dataset.Cabin.isnull()] = 1
dataset['Cabin'].loc[dataset.Cabin.isnull()]=0
# new feature: FamilySize
dataset['FamilySize']= dataset['SibSp'] + dataset['Parch'] + 1
dataset = dataset.drop(['SibSp','Parch'],1)
# fill in empty ages from random forest predictions
dt_age = dataset[['Pclass','Sex','Fare','Cabin','Age']]
dt_age_X = dt_age[dt_age.Age.notnull()].drop('Age',1)
dt_age_y = dt_age[dt_age.Age.notnull()].Age
dt_age_test = dt_age[dt_age.Age.isnull()].drop('Age',1)

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(random_state=0,n_estimators=200)
age_pred = RF.fit(dt_age_X,dt_age_y).predict(dt_age_test)
age_pred = np.round(age_pred).astype(int)

dataset.loc[(dataset.Age.isnull()),'Age'] = age_pred
# continuous values to discrete values
dataset['Age_bin']=pd.cut(dataset['Age'].astype(int),5)
dataset['Fare_bin'] = pd.qcut(dataset['Fare'], 4)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dataset['Age_cde']=label.fit_transform(dataset['Age_bin'])
dataset['Fare_cde']=label.fit_transform(dataset['Fare_bin'])
drop_columns1 = ['Age','Fare','Age_bin','Fare_bin']
dataset = dataset.drop(drop_columns1,1)
# new feature: get titles from names
titles = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0].value_counts()<10
dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
dataset['Title'] = dataset.Title.apply(lambda x: 'Rare' if titles.loc[x]==True else x)
# new feature: get ticket frequency info
dataset['Ticket_cde']=label.fit_transform(dataset['Ticket'])
ticket_vc = dataset.Ticket.value_counts()
ticket_vc[:10]
dataset['Ticket_freq'] = dataset['Ticket'].apply(lambda x: ticket_vc.loc[x])
# new feature: is alone or not
dataset['Is_alone'] = dataset.FamilySize.apply(lambda x: 1 if x==1 else 0)
train2 = pd.concat([train[['Survived']],dataset[:891]],1)
sns.barplot(x='Cabin',y='Survived',data=train2)
sns.barplot(x='FamilySize',y='Survived',data=train2)
sns.barplot(x='Title',y='Survived',data=train2)
sns.barplot(x='Fare_cde',y='Survived',data=train2)
sns.barplot(x='Ticket_freq',y='Survived',data=train2)
sns.barplot(x='Is_alone',y='Survived',data=train2)
drop_columns2 = ['Name','Ticket','Ticket_cde','PassengerId']
dataset = dataset.drop(drop_columns2,1)
dataset['FamilySize_cde'] = dataset.FamilySize.apply(lambda x: 0 if x>=8 else (1 if x<=1 or (x>=5 and x<=7) else 2))
dataset['Title_cde'] = dataset.Title.apply(lambda x: 0 if x=='Mr' else (1 if x=='Master' or x=='Rare' else 2))
dataset['TickeFreq_cde'] = dataset.Ticket_freq.apply(lambda x: 0 if x>=9 else (1 if x<=1 or (x>=5 and x<=8) else 2))
train4 = pd.concat([train[['Survived']],dataset[:891]],1)
sns.barplot(x='FamilySize_cde',y='Survived',data=train4)
sns.barplot(x='Title_cde',y='Survived',data=train4)
sns.barplot(x='TickeFreq_cde',y='Survived',data=train4)
drop_columns3 = ['FamilySize','Title','Ticket_freq']
dataset = dataset.drop(drop_columns3,1)
dataset[:3]
train3 = dataset[:891]
test3 = dataset[891:]
train_y = train[['Survived']]
# model 1: xgboost Classifier  
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train3,train_y)
y_pred = model.predict(test3)
xgb_pred = [round(value) for value in y_pred]

example = pd.read_csv("../input/gender_submission.csv",sep=',',header=0)
data_to_submit = pd.DataFrame({
    'PassengerId':example['PassengerId'],
    'Survived':pd.Series(xgb_pred)
})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
# model 2: Multilayer Perceptron (MLP) 
#         or Artificial Neural Network (ANN)
from sklearn.neural_network import MLPClassifier
MLP_mdl = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
MLP_pred = MLP_mdl.fit(train3, train_y).predict(test3)

data_to_submit = pd.DataFrame({
    'PassengerId':example['PassengerId'],
    'Survived':pd.Series(MLP_pred)
})
data_to_submit.to_csv('MLP_pred_result.csv', index = False)