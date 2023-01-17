# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')





features = ['YearBuilt', 'YrSold']

label = 'SalePrice'
train.head()
X_train = train[features]

y_train = train[label]



model = Pipeline([

                    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),

                    ("std_scaler", StandardScaler()),

                    ("regul_reg", Ridge(alpha=0.05, solver="cholesky")),

                ])

model.fit(X_train, y_train)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



X_test = test[features]

y_pred = model.predict(X_test)

Y_test= train[label]
data = pd.concat([train['SalePrice'], train['YrSold']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=train['YrSold'], y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
data = pd.concat([train['SalePrice'], train['YrSold']], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=train['YrSold'], y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=45);
from sklearn.metrics import mean_absolute_error

tree_clf = DecisionTreeClassifier(max_depth=2)

tree_clf.fit(X_train, y_train)

tree_clf.fit(X_train, y_train)

tree_preds = tree_clf.predict(X_test)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

submission.to_csv('jaydon_cobb_ridge.csv', index=False)