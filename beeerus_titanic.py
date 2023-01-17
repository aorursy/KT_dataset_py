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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
file_name1="/kaggle/input/titanic/train.csv"

file_name2="/kaggle/input/titanic/test.csv"



train=pd.read_csv(file_name1)

test=pd.read_csv(file_name2)
train.head(6)
test.head(6)
train.info()
train.shape
train.isnull().sum()
test.shape
test.isnull().sum()
train.dtypes
test.dtypes
cat_features=[features for features in train.columns if train[features].dtypes=='O']

cat_features
num_features=[features for features in train.columns if train[features].dtypes!='O']

num_features
sns.barplot(x='Pclass',y='Survived',data=train)

plt.title('Survived vs Pclass')
sns.barplot(x='Sex',y='Survived',data=train)
sns.barplot(x='Embarked',y='Survived',data=train)
train.head()
sns.heatmap(train.corr(),annot=True)
Y_train=train['Survived']

train=train.drop('Survived',axis=1)




ntrain = train.shape[0]

ntest = test.shape[0]



data=pd.concat((train,test)).reset_index(drop=True)

data
data['Family_Size']=data['SibSp']+data['Parch']+1
data=data.drop(['SibSp','Parch'],axis=1)
data
data.isnull().sum()
data=data.drop('Cabin',axis=1)
data
data['Title']=data['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)[0]
data['Title'].value_counts()
data['FarePerPerson']=data['Fare']/data['Family_Size']
data.drop(['Ticket','Name'],axis=1,inplace=True)
data
data=data.drop('Fare',axis=1)
cat_feat=[feature for feature in data.columns if data[feature].dtypes=='O']
cat_feat
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data.Sex=le.fit_transform(data.Sex)
data.Embarked=data.Embarked.fillna('mode')
data.isnull().sum()
data.Age=data.Age.fillna(data.Age.median())
data.FarePerPerson=data.FarePerPerson.fillna(data.FarePerPerson.median())

data.isnull().sum()
data['Title']=le.fit_transform(data['Title'])

data['Embarked']=le.fit_transform(data['Embarked'])
data.Age
train = data[:ntrain]

test = data[ntrain:]
train
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
tree=DecisionTreeClassifier()

tree.fit(train,Y_train)
pred=tree.predict(test)

pred
GBC=GradientBoostingClassifier()

GBC.fit(train,Y_train)
pred=GBC.predict(test)

pred
ID = test['PassengerId']
submission=pd.DataFrame()

submission['Survived'] = GBC

submission['PassengerId']=ID

submission.to_csv('submissiongb.csv',index=False)
RFC=RandomForestClassifier()

RFC.fit(train,Y_train)
pred=RFC.predict(test)

pred
submission=pd.DataFrame()

submission['Survived'] = RFC

submission['PassengerId']= ID

submission.to_csv('submissionrf.csv',index=False)