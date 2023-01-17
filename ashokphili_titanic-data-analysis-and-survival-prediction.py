# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
titanic=pd.read_csv('../input/train.csv')
titanic.columns
print(titanic.shape)
print(titanic.head())
titanic.groupby('Embarked').Survived.sum()
titanic.groupby('Embarked').Survived.sum().plot(kind='bar')
titanic.groupby(['Embarked','Pclass']).Survived.sum().reset_index()
titanic[titanic.Survived==1].hist('Age')
titanic[titanic.Survived==1].hist('Fare')
titanic[titanic.Survived==1].groupby('Sex').PassengerId.count().reset_index(name='count').plot(x='Sex',y='count',kind='bar')
print(titanic[titanic.Survived==1].groupby('Fare').PassengerId.count())
titanic.head()
from sklearn.neighbors import KNeighborsClassifier
def convert_sex(x):
    if x=='male':
        return 0
    else:
        return 1
def convert_embark(x):
    if x=='C':
        return 0
    elif x=='Q':
        return 1
    else:
        return 2

titanic['sexconverted']=titanic.Sex.apply(convert_sex).dropna()

titanic['embarkconverted']=titanic.Embarked.apply(convert_embark).dropna()
titanic['Pclass']=titanic.Pclass.dropna()
titanic['Survived']=titanic['Survived'].dropna()
titanic['Age']=titanic['Age'].dropna()
titanic.dropna(inplace=True)
print(titanic.Pclass.isnull().sum())
print(titanic.Survived.isnull().sum())
#titanic['pclassconverted']=titanic.Embarked.apply(convert_pclass)
#titanic.head()
titanic.shape

titanic
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(titanic[['Age','sexconverted','embarkconverted','Pclass']],titanic['Survived'])
predict=neigh.predict(titanic[['Age','sexconverted','embarkconverted','Pclass']])
(predict==titanic['Survived']).sum()/predict.shape[0]

#print(titanic.groupby('Age').PassengerId.count())
#titanic.groupby('Cabin').Survived.sum()
#gender=pd.read_csv('../input/gender_submission.csv')
#gender.columns
# Any results you write to the current directory are saved as output.