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

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
#Data Preprocessing 
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()




train_data.dtypes
train_data.columns
train_data.describe(include='all')
test_data.columns
target= train_data['Survived']

target.shape
train_data.drop(['Survived'], axis=1, inplace = True)
train_data.shape
train1=train_data

test1=test_data



print('Ready to concatinate!')
df = pd.concat([train1, test1], axis=0,sort=False)

df.shape
df.isnull().sum()
PassengerId=test_data.PassengerId

PassengerId.shape
df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

df.shape


df.fillna(df.median(), inplace=True)
df.describe(include='all')
df.isnull().sum()
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.isnull().sum()
df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'SibSp', 'Parch','Embarked'])

df.head()
df_train = df.iloc[:891,:]



df_test = df.iloc[891:,:]



print("Shape of new dataframes - {} , {}".format(df_train.shape, df_test.shape))
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics
df_train = preprocessing.StandardScaler().fit(df_train).transform(df_train)

df_train[0:5]
x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



parameter_grid = {'C': [0.0001, 0.001, 0.01, 0.1], 'gamma' : [00.1, 0.10, 1, 10]}





grid_search = GridSearchCV(SVC(kernel='rbf'), cv=10, param_grid = parameter_grid)



grid_search.fit(x_train, y_train)



print ("Best Score: {}".format(grid_search.best_score_))

print ("Best params: {}".format(grid_search.best_params_))
SVC_model = SVC(C=0.1, gamma=0.1, kernel='rbf')

SVC_model.fit(x_train, y_train)



df_test = preprocessing.StandardScaler().fit(df_test).transform(df_test)

df_test[0:5]
SVC_model.fit(df_train,target)

prediction=SVC_model.predict(df_test)
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")