# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib
url = '../input/crx_data_train_x.csv'
data = pd.read_csv(url, header=None, na_values='?')
#print(data.head())
print(data.columns)


data.columns = ['A'+str(x+1) for x in data.columns]
data.head()
data.describe()
data.describe(include=[object])
data.corr()
data.count()
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
numerical_columns

data = data.fillna(data.median(axis=0), axis=0)
cat_columns   = [c for c in data.columns if data[c].dtype.name == 'object']

desc = data.describe(include=[object])
for c in cat_columns:
    data[c] = data[c].fillna(desc[c]['top'])
data.describe(include=[object])
binary_columns    = [c for c in cat_columns if desc[c]['unique'] == 2]
nonbinary_columns = [c for c in cat_columns if desc[c]['unique'] > 2]
binary_columns, nonbinary_columns
for c in binary_columns:
    top = desc[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
data[binary_columns].describe()
data.describe()
data[nonbinary_columns[0]].unique()
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
data_nonbinary.columns
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
data_numerical.describe()
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
data.shape,data.columns
X = data
url = '../input/crx_data_train_y.csv'
y = pd.read_csv(url, header=None, na_values='?')
y.columns = ['class']
N, d = X.shape
X.shape
url = '../input/crx_data_test_x.csv'
X_test = pd.read_csv(url, header=None, na_values='?')
X_test.head(),X_test.shape
X_test.columns = ['A'+str(x+1) for x in X_test.columns]
X_test.head()
numerical_columns   = [c for c in X_test.columns if X_test[c].dtype.name != 'object']
X_test = X_test.fillna(X_test.median(axis=0), axis=0)
cat_columns   = [c for c in X_test.columns if X_test[c].dtype.name == 'object']

desc = X_test.describe(include=[object])
for c in cat_columns:
    X_test[c] = X_test[c].fillna(desc[c]['top'])

binary_columns    = [c for c in cat_columns if desc[c]['unique'] == 2]
nonbinary_columns = [c for c in cat_columns if desc[c]['unique'] > 2]

for c in binary_columns:
    top = desc[c]['top']
    top_items = X_test[c] == top
    X_test.loc[top_items, c] = 0
    X_test.loc[np.logical_not(top_items), c] = 1

data_nonbinary = pd.get_dummies(X_test[nonbinary_columns])

data_numerical = X_test[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()

X_test = pd.concat((data_numerical, X_test[binary_columns], data_nonbinary), axis=1)
X_test = pd.DataFrame(X_test, dtype=float)
X_test.shape,X_test.columns
X_test = X_test.drop(columns=['A7_o'])
X_test.shape,X_test.columns
X.shape,y.shape
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=10, random_state=11)
rf.fit(X, y.values.ravel())

err_train = np.mean(y.values.ravel() != rf.predict(X))
#err_test  = np.mean(y_test  != rf.predict(X_test))
err_train
from sklearn.svm import SVC
svc = SVC()
svc.fit(X, y.values.ravel())

err_train = np.mean(y.values.ravel() != svc.predict(X))
# err_test  = np.mean(y_test  != svc.predict(X_test))
err_train
svc.predict(X_test)
rf.predict(X_test)
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(n_jobs=-1, random_state=7)
logit.fit(X, y.values.ravel())

err_train = np.mean(y.values.ravel() != svc.predict(X))
# err_test  = np.mean(y_test  != svc.predict(X_test))
err_train
logit.predict(X_test)

