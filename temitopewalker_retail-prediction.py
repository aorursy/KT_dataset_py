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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
features = pd.read_csv('../input/retaildataset/Features data set.csv')
sales = pd.read_csv('../input/retaildataset/sales data-set.csv')
stores = pd.read_csv('../input/retaildataset/stores data-set.csv')
stores
# Let's explore the "feature" dataframe
# Features dataframe contains additional data related to the store, department, and regional activity for the given dates.
# Store: store number
# Date: week
# Temperature: average temperature in the region
# Fuel_Price: cost of fuel in the region
# MarkDown1-5: anonymized data related to promotional markdowns. 
# CPI: consumer price index
# Unemployment: unemployment rate
# IsHoliday: whether the week is a special holiday week or not
# Let's explore the "sales" dataframe
# "Sales" dataframe contains historical sales data, which covers 2010-02-05 to 2012-11-01. 
# Store: store number
# Dept: department number
# Date: the week
# Weekly_Sales: sales for the given department in the given store
# IsHoliday: whether the week is a special holiday week

sales
features['Date'] = pd.to_datetime(feature['Date'])
sales['Date'] = pd.to_datetime(sales['Date'])
features.head()
df = pd.merge(sales, feature, on = ['Store','Date','IsHoliday'])
df = pd.merge(df, stores, on = ['Store'], how = 'left')
sns.heatmap(df.isnull())
df[['year','month','day']] = df.Date.apply(lambda x: pd.Series(x.strftime("%Y,%m,%d").split(",")))
df.drop(['year','day','Date'], axis=1, inplace=True)
df.fillna(0, inplace=True)
df.isnull().sum()
df['month']=df['month'].astype(str).astype(int)
df.IsHoliday = df.IsHoliday.replace({False:0, True:1})

fig = plt.figure(figsize=(16,5))
fig.add_subplot(2,2,1)
sns.countplot(df['IsHoliday'])
fig.add_subplot(2,2,2)
sns.countplot(df['Type'])
df.info()
df_target = df['Weekly_Sales']
df_final = df.drop(['Weekly_Sales'], axis=1)
df_final = pd.get_dummies(df_final, columns = ['Store', 'Dept', 'Type'], drop_first =True)
df_final.isnull().sum()
X = np.array(df_final).astype('float32')
y = np.array(df_target).astype('float32')
y=y.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5)
print('Shape of X_test = ', X_test.shape,  '\nShape of y_test ='  , y_test.shape)
print('Shape of X_train = ', X_train.shape,  '\nShape of y_train ='  , y_train.shape)
print('Shape of X_val = ', X_val.shape,  '\nShape of y_val ='  , y_val.shape)
import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.2, max_depth = 10, n_estimators = 100)
model.fit(X_train, y_train)
# make predictions on the test data

y_predict = model.predict(X_test)
result = model.score(X_test, y_test)

print("Accuracy : {}".format(result))
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
