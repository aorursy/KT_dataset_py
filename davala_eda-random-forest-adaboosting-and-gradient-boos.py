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
# Load the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import pandas_profiling as pf
pd.set_option('display.max_column', 60)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
print('complete')
# load the dataset
data = pd.read_csv('/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')
data.head()
# checking the dimension of the dataset
rows, cols = data.shape
print(f'The number of rows in our dataset are {rows} \nWhile the number of columns are {cols}')
# display the columns
data.columns
# replace space in the columns with underscore
data.columns = data.columns.str.replace(' ', '_')
data.columns
# descriptive statistics
data.describe()
data.select_dtypes('number').head(3)
# lets chech how these attributes correlated to our target variable
data.corr()['churn']
# Lets visualiza our target variable
churn = data['churn'].value_counts()
sns.barplot(churn.index, churn.values)
# dealing with categorical columns 
data.select_dtypes('object').head(4)
# checking the number of values in 'international_plan' columns
data['international_plan'].value_counts()
# checking the number of values in 'voice_mail_plan' columns
data['voice_mail_plan'].value_counts()
# number of unique values in state
data['state'].nunique()
# lets drop phone number columns
data = data.drop(['phone_number'], axis = 1)
# create dummy variables for categorical columns
data = pd.get_dummies(data)
#data
# Target column:
y = data.churn

# features columns
x = data.drop('churn', axis = 1)
# lets train our data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
# instantiate decision tree object with default params
dtc = DecisionTreeClassifier(max_depth = 5)

dtc.fit(x_train, y_train)
pred = dtc.predict(x_test)
accur = accuracy_score(pred, y_test)
print(accur)

# calculate recall_score for test data:
print(precision_score(y_test, dtc.predict(x_test)))

# calculate recall_score for test data:
print(recall_score(y_test, dtc.predict(x_test)))
RF  = RandomForestClassifier(n_estimators = 100, max_depth= 5)
RF.fit(x_train, y_train)

# prediction
pred =RF.predict(x_test)

# model eveluation
print("Accuracy:",accuracy_score(y_test, pred))
print("Precision:",precision_score(y_test, pred))
print("Recall:",recall_score(y_test, pred))
# instantiate adaboost classifier object
ABC = AdaBoostClassifier(random_state = 15)

# fit the model to the training data:
ABC.fit(x_train, y_train)

# prediction
pred =ABC.predict(x_test)

# model eveluation
print("Accuracy:",accuracy_score(y_test, pred))
print("Precision:",precision_score(y_test, pred))
print("Recall:",recall_score(y_test, pred))
# instantiate gradient boost classifier object
GBC = GradientBoostingClassifier(random_state = 15)

# fit the model to the training data:
GBC.fit(x_train, y_train)

# prediction
pred =GBC.predict(x_test)

# model eveluation
print("Accuracy:",accuracy_score(y_test, pred))
print("Precision:",precision_score(y_test, pred))
print("Recall:",recall_score(y_test, pred))