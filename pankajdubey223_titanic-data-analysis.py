# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.#
# i9mporting trainig dataset



titanic_data = pd.read_csv('/kaggle/input/titanic/train_data.csv') 
# importing important library



import seaborn as sns

import math

import matplotlib.pyplot as plt
titanic_data.head()
print('number of passenger in original data :'+ str(len(titanic_data.index)))
#Analyzing data



sns.countplot(x = 'Survived',data = titanic_data)
sns.countplot(x= 'Survived',hue = 'Sex',data = titanic_data)
sns.countplot(x = 'Survived',hue = 'Pclass_1',data = titanic_data)
sns.countplot(x = 'Survived',hue = 'Pclass_2',data = titanic_data)
sns.countplot(x = 'Survived',hue = 'Pclass_3',data = titanic_data)
titanic_data['Age'].plot.hist()
titanic_data['Fare'].plot.hist(bins = 20, figsize = (10,15))
titanic_data.info()
#  Data  Wrangling



titanic_data.isnull().sum()
#sns.heatmap(titanic_data.isnull().sum(),cmap = 'Virdis')
sns.boxplot(x = 'Pclass_1',y = 'Age',data = titanic_data)
sns.boxplot(x = 'Pclass_2',y = 'Age',data = titanic_data)
sns.boxplot(x = 'Pclass_3',y = 'Age',data = titanic_data)
titanic_data.head()
titanic_data.drop(['PassengerId','Unnamed: 0'],axis = 1, inplace = True)
titanic_data.head()
X_train = titanic_data.drop('Survived', axis = 1)

y_train = titanic_data['Survived']
# traninig the data

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(features, lables , test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()



classifier.fit(X_train, y_train)
# importing testing dataset

titanic_data_test = pd.read_csv('/kaggle/input/titanic/test_data.csv') 



titanic_data_test.drop(['PassengerId','Unnamed: 0'],axis = 1, inplace = True)

X_test = titanic_data.drop('Survived', axis = 1)

y_test = titanic_data['Survived']
predictions = classifier.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)