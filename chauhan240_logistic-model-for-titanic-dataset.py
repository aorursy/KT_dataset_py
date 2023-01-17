# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

train.head(30)
train['Embarked'].unique()
train.info()
train.describe()
train.isnull().head()
train.corr()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(train.corr(),annot=True)
#sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Embarked',data=train,palette='Set3')#'RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=True,color='darkred',bins=20)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
train['Fare'].hist(color='green',bins=30,figsize=(8,4))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='summer')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
c=train[['Age','Pclass']]
c.iloc[:,0]
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)

train.head()
train.info()
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)

train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head()
train.info()
from sklearn.model_selection import train_test_split
X = train.drop('Survived',axis=1)

y = train['Survived']

X.head()
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
import pickle
pickle_out=open("classifier.pickle","wb")

pickle.dump(logmodel,pickle_out)

pickle_out.close()
pickle_in=open('classifier.pickle','rb')# rb read mode

cnn_classifier=pickle.load(pickle_in)
cnn_classifier.predict(X_test)
predictions = logmodel.predict(X_test)
predictions
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions,digits=4))
print(confusion_matrix(y_test,predictions))
logmodel.coef_