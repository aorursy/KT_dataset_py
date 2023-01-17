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
train=pd.read_csv('/kaggle/input/datasetforspamdetection/Train.csv')

test=pd.read_csv('/kaggle/input/datasetforspamdetection/TestX.csv')
train
test
train.head()
train.tail()
test.head()
test.tail()
train.isnull().sum()
test.isnull().sum()
train.info()
test.info()
output=pd.read_csv('/kaggle/input/datasetforspamdetection/output.csv')
output
output.head()
output.tail()
output.isnull().sum()
output.info()
from sklearn.model_selection import train_test_split

train,test=train_test_split(output,test_size=0.1,random_state=1)
def data_splitting(df):

    x=df.drop(['Id'],axis=1)

    y=df['Id']

    return x,y



x_train,y_train=data_splitting(train)

x_test,y_test=data_splitting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

prediction=log_model.predict(x_test)

score=accuracy_score(y_test,prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train)

print(reg_test)
def data_splitting(df):

    x=df.drop(['Label'],axis=1)

    y=df['Label']

    return x,y



x_train,y_train=data_splitting(train)

x_test,y_test=data_splitting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

prediction=log_model.predict(x_test)

score=accuracy_score(y_test,prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train)

print(reg_test)
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('Label', data=output)

plt.title('Graph', fontsize=16)

plt.ylabel("Id")

plt.show()
sns.lineplot(x=output.Label,y=output.Id)
sns.scatterplot(x=output.Label,y=output.Id)
sns.boxplot(x=output.Label,y=output.Id)
sns.swarmplot(x=output.Label,y=output.Id)
sns.stripplot(x=output.Label,y=output.Id)
sns.jointplot(x=output.Label,y=output.Id)
sns.distplot(output.Id.dropna(),bins=8,color='Green')
sns.distplot(output.Label.dropna(),bins=8,color='Blue')