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
import seaborn as sns

import numpy as np # linear algebra

import pandas as pd 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

import math

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
filepath= '/kaggle/input/amazon-employee-access-challenge/train.csv'

traindata= pd.read_csv(filepath)
filepath2= '/kaggle/input/amazon-employee-access-challenge/test.csv'

testdata= pd.read_csv(filepath2)

testdatacopy=testdata

traindata.head()
sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')
traindata
testdata
testdata=testdata.drop('id',axis=1)
y=traindata['ACTION']

x=traindata.drop('ACTION',axis=1)

#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
#Logistic Regression

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x_train, y_train)

y_predicted = LogisticRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

print('f1 score:')

print(f1_score(y_test,y_predicted))

print('accuracy score:')

print(accuracy_score(y_test,y_predicted))
# Random Forest

rf = RandomForestClassifier()

rf.fit(x_train,y_train);

y_predicted_r = rf.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_r)

r = r2_score(y_test, y_predicted_r)

mae = mean_absolute_error(y_test,y_predicted_r)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

print('f1 score:')

print(f1_score(y_test,y_predicted_r))

print('accuracy score:')

print(accuracy_score(y_test,y_predicted_r))
predictions = rf.predict(testdata)
predictions
predictionlist=y_predicted.tolist()
predictionlist
predix  =testdatacopy['id'].tolist()
predix
output=pd.DataFrame(list(zip(predix  , predictionlist)),

              columns=['id','ACTION'])



output.to_csv('my_submission.csv', index=False)
output