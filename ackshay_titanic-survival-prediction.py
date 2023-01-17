# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

train=pd.read_csv('../input/titanic_train.csv')
train.head(2)
train.shape
df=pd.DataFrame(train)

plt.subplots(figsize=(25,15))

plt.xticks(rotation=90)



sns.set(style="whitegrid")

g=sns.countplot(x="Age",hue="Survived",data=train)

sns.countplot(x="Age",hue="Sex",data=train,color="red")
sns.barplot(x="Sex",y="Survived",data=train)
plt.subplots(figsize=(14,10))

sns.stripplot(x="Sex",y="Age",data=train,jitter=True,hue="Survived")
sns.countplot(x='Embarked',data=train,hue="Survived")

sns.countplot(x='Pclass',hue='Survived',data=train)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train)
def impute(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

    

        elif Pclass==2:

            return 29

    

        else:

            return 24

    else:

        return Age

        
train['Age']=train[['Age','Pclass']].apply(impute,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")
del train['Cabin']
train.dropna(inplace=True)

train.shape
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sex=pd.get_dummies(train['Sex'],prefix='Sex',drop_first=True)

Pclass=pd.get_dummies(train['Pclass'],prefix="Pclass",drop_first=True)

embarked=pd.get_dummies(train['Embarked'],prefix="Embarked",drop_first=True)

train=train.join([sex,Pclass,embarked])
train.head(2)
train.drop(['Embarked','Fare','Pclass','SibSp','Ticket','Name','Parch','PassengerId','Sex'],inplace=True,axis=1)
train.head()


full_x=train.drop('Survived',axis=1)
full_y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(full_x,full_y,test_size=30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))