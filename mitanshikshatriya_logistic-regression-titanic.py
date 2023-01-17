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
filename = '/kaggle/input/titanic/train.csv'

df = pd.read_csv(filename)

df.head()
df.shape
df.describe()
#list of columns

df.columns
#checking for null values

df.isnull().sum().sort_values(ascending=False)
#droping cabin column from the dataset as more than 50% of it is null values

df=df.drop(columns=['Cabin'])

df
import matplotlib.pyplot as plt

import seaborn as sns
sns.heatmap(df.corr(),annot=True)
sns.jointplot(x="Age", y="Pclass", data=df,kind="hex",height=8,color="g") 

plt.show()
#using median value grouped by pclass to replace missing values

age_group = df.groupby('Pclass')["Age"]

age_group.median()
#replacing null age values using median grouped by Pclass

df.loc[df.Age.isnull(),'Age']=df.groupby('Pclass').Age.transform('median')

print(df.Age.isnull().sum())
from statistics import mode

#filling embarked with mode value as only 2 values of embarked are missing

df['Embarked'] = df.Embarked.fillna(mode(df['Embarked']))

df.isnull().sum()
sns.countplot('Survived',data=df)
sns.countplot('Sex',hue='Survived',data=df)
sns.barplot(x='Pclass',y='Survived',data=df)
sns.barplot(x='Embarked',y='Survived',hue='Sex',data=df)
# https://www.kaggle.com/konohayui/titanic-data-visualization-and-models



#df.loc[df['Survived']==1,'Age']



sns.distplot(a=df.loc[df['Survived']==1,'Age'])

sns.distplot(a=df.loc[df['Survived']==0,'Age'])

plt.legend(labels=['Survived','dead'])

plt.show()
sns.distplot(a=df.loc[df['Survived']==1,'Fare'])

sns.distplot(a=df.loc[df['Survived']==0,'Fare'])

plt.legend(labels=['Survived','dead'])

plt.show()
#dividing age into bands

df.AgeCut=pd.cut(df.Age,5)

print(df.AgeCut)

df.AgeCut.value_counts().sort_index()
# replace agebands with ordinals

df.loc[df.Age<=16.136,'AgeCut']=1

df.loc[(df.Age>16.136)&(df.Age<=32.102),'AgeCut']=2

df.loc[(df.Age>32.102)&(df.Age<=48.068),'AgeCut']=3

df.loc[(df.Age>48.068)&(df.Age<=64.034),'AgeCut']=4

df.loc[df.Age>64.034,'AgeCut']=5
df.FareCut=pd.qcut(df.Fare,5)

df.FareCut.value_counts().sort_index()



df.loc[df.Fare<=7.854,'FareCut']=1

df.loc[(df.Fare>7.854)&(df.Fare<=10.5),'FareCut']=2

df.loc[(df.Fare>10.5)&(df.Fare<=21.558),'FareCut']=3

df.loc[(df.Fare>21.558)&(df.Fare<=41.579),'FareCut']=4

df.loc[df.Fare>41.579,'FareCut']=5

#df['FareCut']
#label encoding Sex column

df.loc[df.Sex=='male','Sex']=0

df.loc[df.Sex=='female','Sex']=1

df
df.loc[df.Sex=='male','Sex']=0

df.loc[df.Sex=='female','Sex']=1

df.loc[df.Embarked=='C','Embarked']=0

df.loc[df.Embarked=='S','Embarked']=1

df.loc[df.Embarked=='Q','Embarked']=2

df.Embarked.value_counts()
df.columns.values
X=df[['Sex','AgeCut','SibSp','Parch','FareCut','Embarked','Pclass']]

y=df[['Survived']]

X.head()



#y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01,solver='lbfgs')

LR.fit(X_train,y_train)

LR
yhat=LR.predict(X_test)

print(yhat[0:5],y_test[0:5])

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss

LR_f1= f1_score(y_test, yhat, average='weighted')

print ("Logistic Regression f1 score: %.2f" % LR_f1)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR1 = LogisticRegression(C=0.01,solver='lbfgs')

LR1.fit(X,y)

LR1
test_filename='/kaggle/input/titanic/test.csv'

test = pd.read_csv(test_filename)

test.head()
test.isnull().sum()
test.loc[test.Age.isnull(),'Age']=test.groupby('Pclass').Age.transform('median')

test['Fare'] = df.Fare.fillna(mode(test['Fare']))



test.isnull().sum()
#age

test.loc[test.Age<=16.136,'AgeCut']=1

test.loc[(test.Age>16.136)&(test.Age<=32.102),'AgeCut']=2

test.loc[(test.Age>32.102)&(test.Age<=48.068),'AgeCut']=3

test.loc[(test.Age>48.068)&(test.Age<=64.034),'AgeCut']=4

test.loc[test.Age>64.034,'AgeCut']=5

test.AgeCut

#fare

test.loc[test.Fare<=7.854,'FareCut']=1

test.loc[(test.Fare>7.854)&(test.Fare<=10.5),'FareCut']=2

test.loc[(test.Fare>10.5)&(test.Fare<=21.558),'FareCut']=3

test.loc[(test.Fare>21.558)&(test.Fare<=41.579),'FareCut']=4

test.loc[test.Fare>41.579,'FareCut']=5
test.loc[test.Sex=='male','Sex']=0

test.loc[test.Sex=='female','Sex']=1

test.loc[test.Embarked=='C','Embarked']=0

test.loc[test.Embarked=='S','Embarked']=1

test.loc[test.Embarked=='Q','Embarked']=2

test.Embarked.value_counts()
X_test=test[['Sex','AgeCut','SibSp','Parch','FareCut','Embarked','Pclass']]

X.head()
yhat1=LR1.predict(X_test)

yhat1
outcome = pd.DataFrame(yhat1)

pass_id = pd.read_csv(test_filename)[['PassengerId']]

result = pd.concat([pass_id,outcome], axis=1)

result.columns = ['PassengerId','Survived']

result.to_csv('result.csv',encoding='utf-8', columns=['PassengerId','Survived'], index=False)
