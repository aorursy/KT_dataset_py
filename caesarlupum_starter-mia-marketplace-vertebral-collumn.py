import seaborn as sns

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
nRowsRead = 1000 # specify 'None' if want to read whole file

# column_2C.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/column_2C.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'column_2C.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# column_3C.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/column_3C.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'column_3C.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
sns.pairplot(df2, hue="class", size=3, diag_kind="kde")
df2['class'] = df2['class'].map({'Normal': 0, 'Hernia': 1, 'Spondylolisthesis': 2})

df1['class'] = df1['class'].map({'Normal': 0, 'Abnormal': 1})

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.metrics import accuracy_score

X = df1[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope', 'pelvic_radius','degree_spondylolisthesis']]

Y = df1['class']

# split data into train and test sets

seed = 2020

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

import xgboost as xgb



# fit model no training data

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# save model to file

model.save_model("model.bst")
# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_test.shape, X_test.shape
X_test.min()
X_test.max()
X_test.mean()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# CV model

model = xgb.XGBClassifier()

kfold = KFold(n_splits=10, random_state=2020)

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
xgb.__version__
from platform import python_version



print(python_version())