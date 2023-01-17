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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
parameter_grid = {'bootstrap': [True],
                    'max_depth': [80, 90, 100, 110],
                    'max_features': [2, 3],
                     'min_samples_leaf': [3, 4, 5],
                     'min_samples_split': [8, 10, 12],
                        'n_estimators': [100, 200, 300]}

grid_search = GridSearchCV(model, param_grid = parameter_grid,
                          cv =10)

grid_search.fit(x_train, y_train)

print ("Best Score: {}".format(grid_search.best_score_))
print ("Best params: {}".format(grid_search.best_params_))
model = RandomForestClassifier(bootstrap=True, max_depth= 110, max_features= 2, min_samples_leaf= 3, min_samples_split= 8, n_estimators= 100)
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import classification_report,log_loss,f1_score, accuracy_score
print(classification_report(y_test,pred))
print('\n')
print('F1-SCORE : ',f1_score(y_test,pred,average=None))
print('\n')
print('Train Accuracy: ', accuracy_score(y_train, model.predict(x_train))*100,'%')

df_test = preprocessing.StandardScaler().fit(df_test).transform(df_test)
df_test[0:5]
model.fit(df_train,target)
prediction=model.predict(df_test)
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")