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



# Any results you write to the current directory are saved as output.
# Read Mushroom CSV Data

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv', 

                 usecols=["class", 'odor', 'population'], header=0)



df.head()
df.odor.unique()
# Check for Null inputs 

df.isnull().sum() # No Null values in the dataset
df.shape 
print(df.groupby('odor').size())
# Preprocess values and convert it to unique integers

from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()

for col in df.columns:

    df[col] = labelencoder.fit_transform(df[col])

 

df.head()
from sklearn.model_selection import train_test_split



# 6500 Training inputs, 1625 Testing inputs

X_train, X_test, y_train, y_test = train_test_split(df, df.odor, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Standardize the data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the training set and the test set.

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train, X_test)

print(X_test.min(),X_test.max())
from sklearn.linear_model import LogisticRegression

# Since this problem is a multiclass classification problem, use logistic regression

logRegModel = LogisticRegression(max_iter=8124).fit(X_train, y_train)
logRegModel.coef_
logRegModel.intercept_
from sklearn import metrics



# use the model to make predictions with the test data

# how did our model perform?

y_pred = logRegModel.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logRegModel.score(X_test, y_test)))
# Check for correctness before checking accuracy

print(y_pred[0:10])

print(y_test[0:10])
from matplotlib import pyplot as plt



## Plot the prediction vs true value

plt.scatter(y_test, y_pred)

plt.xlabel('True Values')

plt.ylabel('Predictions')
# Thus, now given an input you can predict the mushroom type

# Since model only takes integer or floats as input, given those input now the model can predict the mushroom.

prediction = logRegModel.predict([[ 2, 1 ,0]])

print(prediction)