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
df = pd.read_csv("/kaggle/input/covid19/increment sales data filtered.csv")
df.shape
df.head()
df['delta'] = df['base_forecast_units'] - df['Units'] 
df.columns
to_drop = ['base_forecast_units', 'Units', 'BC Department', 'BC Super Category', 'BC Category',
       'BC Sub Category', 'Weeks', 'Continued Claims', 'Covered Employment']
df = df.drop(to_drop, axis=1)
df.columns
df['SIZE'] = df['BASE SIZE'].str[0:2]
df = df.drop(['BASE SIZE'],axis=1)
df['SIZE'] = df['SIZE'].astype('float')
df = df.drop(['Initial Claims', 'date'], axis=1)
dfd = pd.get_dummies(df)
dfd.head()
dfd = dfd[~dfd['delta'].isna()]
y = dfd['delta']
X = dfd.drop(['delta'], axis=1)
from sklearn.model_selection import train_test_split
X = X.fillna(0.0)

Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import Lasso, LassoCV
reg = LassoCV(cv=5, random_state=0).fit(Xtr, ytr)
reg.score(Xtr, ytr)
reg.get_params
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtr, ytr)

from sklearn.metrics import r2_score, mean_absolute_error
print(r2_score(ytest, lr.predict(Xtest)))
print(mean_absolute_error(ytest, lr.predict(Xtest)))
