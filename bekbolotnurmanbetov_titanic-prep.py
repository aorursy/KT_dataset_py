# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/titanic/train.csv")
df.head()

df['Sex'].value_counts()
sns.countplot(df['Sex'])

df['Pclass'].value_counts()
df.groupby(['Pclass','Sex']).size()
sns.factorplot('Pclass',data=df,hue='Sex',kind='count')
def child(x):
    age,sex=x
    if age<16:
        return 'child'
    else:
        return sex        
df['Person']=df[['Age','Sex']].apply(child,axis=1)
df.head(10)
sns.factorplot('Pclass',data=df,kind='count',hue='Person')
sns.factorplot('Sex',data=df,kind='count',hue='Pclass')
df['Age'].hist()
def fam(x):
    sb,p=x
    if (sb==0) & (p==0):
        return 'alone'
    else:
        return 'family'
    
df['Family']=df[['SibSp','Parch']].apply(fam,axis=1)
df.head()
sns.factorplot('Family',data=df,kind='count',hue='Pclass')

sns.factorplot('Family',data=df,kind='count',hue='Sex')

df[df['Sex']=='female'].Age.hist()

df[df['Sex']=='female'].Age.mean()

df[df['Sex']=='male'].Age.hist()

df[df['Sex']=='male'].Age.mean()

fig=sns.FacetGrid(df,hue='Pclass',aspect=3)
fig.map(sns.kdeplot,'Age',shade=True)
a=df['Age'].max()
fig.set(xlim=(0,a))
fig.add_legend()
sns.countplot(df['Survived'])
df.isnull().sum()
df=df[[ 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Embarked', 'Person', 'Family']]
df.isnull().sum()
df['Embarked'].fillna('S',inplace=True)
df['Age'].interpolate(inplace=True)
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
x=['Survived', 'Pclass','Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Embarked', 'Person', 'Family']
for i in x:
    a=LabelEncoder()
    df[i]=a.fit_transform(df[i])
df.head()
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
s=minmax.fit_transform(df[['Age','Fare']])
df[['Age','Fare']]=pd.DataFrame(s)
df.head()
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(df[['Pclass','Sex', 'Age', 'SibSp',
       'Parch',  'Fare', 'Embarked', 'Person', 'Family']],df['Survived'],test_size=0.3)
trainX.shape
import xgboost as xgb
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)

train_model1 = model1.fit(trainX, trainY)
train_model2 = model2.fit(trainX, trainY)
from sklearn.metrics import accuracy_score
pred1 = train_model1.predict(testX)
pred2 = train_model2.predict(testX)
print("Accuracy for model 1: %.2f" % (accuracy_score(testY, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(testY, pred2) * 100))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc_model = rfc.fit(trainX, trainY)
pred3 = rfc_model.predict(testX)
print("Accuracy for Random Forest Model: %.2f" % (accuracy_score(testY, pred3) * 100))
