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
# Imports

from mlxtend.preprocessing import minmax_scaling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score , mean_squared_error

import warnings

warnings.filterwarnings("ignore")
# Read data

dataframe = pd.read_csv('../input/heart-disease-uci/heart.csv' , engine = 'c')

dataframe
# Scale the column 'age'

dataframe['scaled_age'] = minmax_scaling( dataframe , columns = 'age' )

dataframe
# Split Train and Test

features = dataframe.iloc[ : , :-2]

target = dataframe.iloc[ : , -2]

x_train, x_test, y_train, y_test = train_test_split(features , target , test_size=0.2, random_state=0)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
knn_model = KNeighborsClassifier(n_neighbors=2)

knn_model.fit(x_train , y_train)

y_pred = knn_model.predict(x_test)
# Accuracy

accuracy_score(y_test, y_pred)
# MSE

mean_squared_error(y_test, y_pred)
# Logistic Regression Model

log_regr_model = LogisticRegression(random_state=0)

log_regr_model.fit(x_train , y_train)

y_pred = log_regr_model.predict(x_test)
# Accuracy

accuracy_score(y_test, y_pred)
# MSE

mean_squared_error(y_test, y_pred)