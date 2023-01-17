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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
traindf=pd.read_csv('/kaggle/input/predict-red-wine-quality/train.csv')

testdf=pd.read_csv('/kaggle/input/predict-red-wine-quality/test.csv')
testdf.head()
traindf.head()
traindf.isnull().sum()
testdf.isnull().sum()
traindf.info()
traindf.describe()
sns.jointplot('density', 'quality', data = traindf, kind="reg")

plt.show()
X = traindf.drop("quality", axis=1)

X.head()
y = traindf[['quality']]

y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8, random_state=48)
X_train.shape, X_test.shape
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
X_train.columns
lr1.coef_
from sklearn.metrics import r2_score
y_train_pred = lr1.predict(X_train)
r2_score(y_train, y_train_pred)
corrs = X_train.corr()
plt.figure(figsize=[10,8])

sns.heatmap(corrs, cmap="Reds", annot=True)

plt.show()
from sklearn.feature_selection import RFE
lr2 = LinearRegression()
rfe_selector = RFE(lr2, 10, verbose=True)
rfe_selector.fit(X_train, y_train)
rfe_selector.support_
rfe_selector.ranking_
cols_keep = X_train.columns[rfe_selector.support_]

cols_keep
lr2 = LinearRegression()
lr2.fit(X_train[cols_keep],y_train)
y_train_pred = lr2.predict(X_train[cols_keep])
r2_score(y_train, y_train_pred)
lr3 = LinearRegression()
lr3.fit(X_test,y_test)
lr3.coef_