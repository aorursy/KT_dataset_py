# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split # utils
from sklearn.metrics import mean_absolute_error # eval metric

# data processing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNet # machine learning

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# using pandas to read CSV files
df = pd.read_csv('../input/X_train.csv')
y = pd.read_csv('../input/y_train.csv')['PRP']
df_test = pd.read_csv('../input/X_test.csv')
df.head()
fig, ax = plt.subplots(3,2,figsize=(10,8))
for i,c in enumerate(df):
    sns.distplot(df[c], ax=ax[i // 2][i % 2], kde=False)
fig.tight_layout()
sns.distplot(y)
print(y.head())
# split the data
X, y = df.values, y.values
X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=2018)

# data preprocessing using sklearn Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)), # multiply features together
    ('scale', StandardScaler()), # scale data
])

# fit and apply transform
X_train = pipeline.fit_transform(X_train)
# transform the validation set
X_val = pipeline.transform(X_val)
print('train shape:', X_train.shape, 'validation shape:', X_val.shape)
reg = ElasticNet(alpha=1.7)
reg.fit(X_train, y_train) # magic happens here
y_pred = reg.predict(X_val)
y_pred[y_pred < 0] = 0
print('Model MAE:', mean_absolute_error(y_val, y_pred))
print('Mean  MAE:', mean_absolute_error(y_val, np.full(y_val.shape, y.mean())))
# refit and predict submission data
X_train = pipeline.fit_transform(X)
X_test = pipeline.transform(df_test.values)
reg.fit(X_train, y)
y_pred = reg.predict(X_test)
y_pred[y_pred < 0] = 0

df_sub = pd.DataFrame({'Id': np.arange(y_pred.size), 'PRP': y_pred})
df_sub.to_csv('submission.csv', index=False)