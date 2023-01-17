#LOGISTIC VS RANDOIMFOREST VS DECISIONTREES

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TrainDataset=pd.read_csv("../input/train.csv")
TrainDataset.isnull().sum()
sns.heatmap(TrainDataset.isnull(),yticklabels=False)
sns.set_style("whitegrid")

sns.countplot(x=TrainDataset.Survived,hue='Sex',data=TrainDataset)
sns.set_style("whitegrid")

sns.countplot(x=TrainDataset.Survived,hue='Pclass',data=TrainDataset,palette='rainbow')
sns.distplot(TrainDataset['Age'],kde=False,color='darkred',bins=30)
plt.hist(x=TrainDataset['Age'],bins=30,color='darkred',alpha=0.8)
print("We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation). However we can be smarter about this and check the average age by passenger class. For example:")
plt.subplots(figsize=(10,10))

sns.boxplot(x=TrainDataset['Pclass'],y=TrainDataset['Age'])
print("We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.")
CLASS1AGE = TrainDataset.query('Pclass==1')

CLASS2AGE = TrainDataset.query('Pclass==2')

CLASS3AGE = TrainDataset.query('Pclass==3')
CLASS1AGEAVERGAE = CLASS1AGE['Age'].mean()

CLASS2AGEAVERGAE = CLASS2AGE['Age'].mean()

CLASS3AGEAVERGAE = CLASS3AGE['Age'].mean()
CLASS3AGEAVERGAE
TrainDataset.isnull().sum()
def AgeNullReplcaeFunction(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return CLASS1AGEAVERGAE

        elif Pclass==2:

            return CLASS2AGEAVERGAE

        else:

            return 24

    else:

        return Age
TrainDataset['Age'] = TrainDataset[['Age','Pclass']].apply(AgeNullReplcaeFunction,axis=1)
TrainDataset.isnull().sum()
sns.heatmap(TrainDataset.isnull(),yticklabels=False)
from sklearn.preprocessing import LabelEncoder

LabelEmbarked=LabelEncoder()
TrainDataset['Embarked']=TrainDataset['Embarked'].replace(np.nan,'S')
LabelEmbarked.fit(TrainDataset.Embarked)
TrainDataset.loc[TrainDataset['Embarked'].isnull()]
TrainDataset.Embarked=LabelEmbarked.transform(TrainDataset.Embarked)
TrainDataset.drop(['Name','Ticket'],axis=1,inplace=True)
TrainDataset.shape

from sklearn.preprocessing import LabelEncoder

SexEncoder=LabelEncoder()

SexEncoder.fit(TrainDataset.Sex.unique())

TrainDataset.Sex=SexEncoder.transform(TrainDataset.Sex)
Indendent = ['PassengerId','Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked']

Xloc=TrainDataset[Indendent]

Yloc=TrainDataset['Survived']

from sklearn.model_selection import train_test_split

XTrain,XTest,YTrain,YTest= train_test_split(Xloc,Yloc,test_size=0.20,random_state=0)
XTrain.shape
#LogisticRegression

from sklearn.linear_model import LogisticRegression

LogisClassfier = LogisticRegression()

LogisClassfier.fit(XTrain,YTrain)

YPred=LogisClassfier.predict(XTest)

from sklearn.metrics import accuracy_score,confusion_matrix

CM= confusion_matrix(YTest,YPred)

CM
AS=accuracy_score(YTest,YPred)

AS
#RandomForestRegression

from sklearn.ensemble import RandomForestRegressor

RandomForestClassfier = RandomForestRegressor(n_estimators=100)

RandomForestClassfier.fit(XTrain,YTrain)
YRandomPred=RandomForestClassfier.predict(XTest)
for i in range(len(YRandomPred)):

    if (YRandomPred[i]>0.5):

        YRandomPred[i]=1

    else:

        YRandomPred[i]=0
YRandomPred
from sklearn.metrics import accuracy_score,confusion_matrix

CMRandom= confusion_matrix(YTest,YRandomPred)
CMRandom
AS=accuracy_score(YTest,YRandomPred)

AS
#DecisionTreeClassfier

from sklearn.tree import DecisionTreeClassifier

TreeClassfier = DecisionTreeClassifier()

TreeClassfier.fit(XTrain,YTrain)
YPredTreeClassfier= TreeClassfier.predict(XTest)
YPredTreeClassfier
from sklearn.metrics import accuracy_score,confusion_matrix

CMRandom= confusion_matrix(YTest,YPredTreeClassfier)
CMRandom
AS=accuracy_score(YTest,YPredTreeClassfier)

AS