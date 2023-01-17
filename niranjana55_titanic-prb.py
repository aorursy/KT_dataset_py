# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data=pd.read_csv('../input/train.csv')

data.head()
import pandas as pd

# Varies with sklearn version

from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.metrics import mean_absolute_error

from sklearn.externals import joblib



# Load the data set

df = pd.read_csv("../input/train.csv")



# Remove the fields from the data set that we don't want to include in our model

del df['Ticket']

del df['Cabin']

del df['Fare']

del df['Embarked']

del df['Sex']

del df['Name']



# Replace categorical data with one-hot encoded data

features_df = pd.get_dummies(df, columns=['SibSp'])



# Remove the from the feature data

del features_df['Survived']



# Create the X and y arrays

X = features_df.as_matrix()

y = df['Survived'].as_matrix()



X,y
features_df.to_csv('SampleX.csv')

df['Survived'].to_csv('Sampley.csv')
# Split the data set in a training set (70%) and a test set (30%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



X_train, X_test, y_train, y_test 
len(data),data.PassengerId.max()
data.count()
data['Survived'].value_counts()
data['Survived'].value_counts()*100/len(data)
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=data,ax=ax[1])

ax[1].set_title('Survive')

plt.show()
data['Sex'].value_counts()
data[['Sex','Survived']].groupby(['Sex']).mean()
import pandas as pd

import numpy as np

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
f,ax=plt.subplots(1,2,figsize=(18,8))

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
data['Pclass'].value_counts()
pd.crosstab(data.Pclass,data.Survived,margins=True)
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r')
cm = sns.light_palette("green", as_cmap=True)



pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap=cm)
f,ax=plt.subplots(1,2,figsize=(18,8))

data['Pclass'].value_counts().sort_index().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab([data.Sex,data.Pclass],data.Survived,margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived',hue='Sex',data=data)

plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)

plt.show()
data['Age'].min(),data['Age'].max()
%matplotlib inline



alpha_color=0.5

data['Survived'].value_counts().plot(kind='bar')





data['Sex'].value_counts().plot(kind='bar',

                               color=['b','r'],

                               alpha=alpha_color)
data[data['Survived']==1]['Age'].value_counts().sort_index().plot('bar')
bins=[0,10,20,30,40,50,60,70,80]

data['AgeBin']=pd.cut(data['Age'],bins)

data.head()
data[data['Survived']==1]['AgeBin'].value_counts().sort_index().plot('bar')
data[data['Survived']==0]['AgeBin'].value_counts().sort_index().plot('bar')
f,ax=plt.subplots(1,2,figsize=(20,10))

data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
data.groupby('Embarked').count()
data['Embarked'].fillna('S',inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

#data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
data['Gender']=data['Sex'].apply(lambda a: 1 if a=='male' else 0)



data=data.drop(['PassengerId','Name','Cabin','Embarked','Ticket','Sex'],axis=1)



data=data.dropna()



data.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)
data=pd.read_csv('../input/train.csv')

data.head()
data1=pd.read_csv('../input/test.csv')

data1.head()
clf
import pandas as pd

import numpy as np

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

datafile = "titanic.csv"

data = pd.read_csv(datafile)



data.head()


Titanic=pd.read_csv('../input/train.csv')

Titanic
Titanic=pd.read_csv('../input/test.csv')

Titanic
Titanic.head()
Titanic.groupby(['Sex']).count()

names=Titanic[Titanic['Age']<10]

names
sibsp=Titanic[Titanic['SibSp']>=1]

sibsp
free=Titanic[Titanic['Fare']==0]

free
people=Titanic['Embarked'].value_counts()

people
#Show the number passengers survived based on

#a) Age

v1=Titanic['Age'].value_counts()

v1

#b) Gender

v2=Titanic['Sex'].value_counts()

v2



#c) Pclass

v3=Titanic['Pclass'].value_counts()

v3

#d) Embarked

v4=Titanic['Embarked'].value_counts()

v4



v5=v1,v2,v3,v4

v5
Titanic_index=Titanic.set_index(['Pclass','Sex'])

Titanic

v1=Titanic_index.loc[1,'male']

v1['Age'].mean()
Titanic1=Titanic.sort_values('Fare',ascending=False)

Titanic1[:3]
Titanic1['Fare'].max()
len(Titanic),Titanic.PassengerId.max()
Titanic.count()