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
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
df_train = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')
df_train.shape
df_train.head(5)
df_test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
df_test.shape

ids = df_test['id']
df_test.head()

zero_deviation = []
for col in df_train.columns:
    if(df_train[col].std() == 0.0):
        zero_deviation.append(col)
zero_deviation.append('time')
zero_deviation.append('id')
zero_deviation

df_train = df_train.drop(zero_deviation,axis=1)
df_test = df_test.drop(zero_deviation,axis=1)
df_train.shape
df_train.head()
df_test.shape
df_test.head()
df_test.head()
X_train = df_train.iloc[:,:-1]
Y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,:]
X = X_train.append(X_test)

scaler = RobustScaler()
X = scaler.fit_transform(X)
X.shape
X_train = X[:161168]
X_test = X[161168:]
print(X_train.shape)
print(X_test.shape)

X_train_new, X_valid_new,y_train_new,y_valid_new = train_test_split(X_train,Y_train,test_size=0.1,shuffle=False)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train_new,y_train_new)
y_pred = dt.predict(X_valid_new)
sqrt(mean_squared_error(y_valid_new, y_pred))
2.020429201056235
regr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=100)
regr.fit(X_train,Y_train)
1.216292231672738
1.228547551907994

1.1711313363841251

y_pred = regr.predict(X_valid_new)
sqrt(mean_squared_error(y_valid_new, y_pred))
bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=200, random_state=0)
bag.fit(X_train_new,y_train_new)
y_pred = bag.predict(X_valid_new)
sqrt(mean_squared_error(y_valid_new, y_pred))
regr3 = HistGradientBoostingRegressor(max_iter=10000)
regr3.fit(X_train_new,y_train_new)
y_pred = regr3.predict(X_valid_new)
sqrt(mean_squared_error(y_valid_new, y_pred))
cm = cat.CatBoostRegressor(iterations=1000, learning_rate=0.5, depth=12, random_seed=2019)
cm.fit(X_train_new,y_train_new)
y_pred = cm.predict(X_valid_new)
sqrt(mean_squared_error(y_valid_new, y_pred))
y_pred_test = regr.predict(X_test)
df_output = pd.DataFrame({'id':ids,'label':y_pred_test})
df_output.shape
from IPython.display import HTML
import base64
def create_download_link(df, title = "Download CSV file", filename = "ml_out_final.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe
create_download_link(df_output)
