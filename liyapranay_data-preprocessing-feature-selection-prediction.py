# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv')

df_feature = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv')

df_store = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')

df_test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv')
df_train.head()
df_feature.head()
df_store.head()
df_train['Date'] =pd.to_datetime(df_train['Date'], format="%Y-%m-%d")

df_feature['Date'] =pd.to_datetime(df_feature['Date'], format="%Y-%m-%d")

df_test['Date'] = pd.to_datetime(df_test['Date'], format="%Y-%m-%d")
combined_train = pd.merge(df_train,df_store,how='left',on='Store')

combined_test = pd.merge(df_test,df_store,how='left',on='Store')

combined_train = pd.merge(combined_train, df_feature, how = "inner", on=["Store","Date",'IsHoliday'])

combined_test = pd.merge(combined_test, df_feature, how = "inner", on=["Store","Date",'IsHoliday'])

combined_train.head()
combined_train.fillna(0,inplace=True)
combined_train.head()
combined_train.describe()
combined_train['Weekly_Sales'][combined_train['Weekly_Sales'] < 0] = 0

combined_train['MarkDown2'][combined_train['MarkDown2'] < 0] = 0

combined_train['MarkDown3'][combined_train['MarkDown3'] < 0] = 0
sns.set(rc={'figure.figsize':(10,8)})

sns.scatterplot(combined_train['Size'],combined_train['Weekly_Sales'],hue=combined_train['IsHoliday']);
sns.scatterplot(combined_train['Fuel_Price'],combined_train['Weekly_Sales']);
sns.scatterplot(combined_train['MarkDown1'],combined_train['Weekly_Sales']);
sns.scatterplot(combined_train['MarkDown2'],combined_train['Weekly_Sales']);
sns.scatterplot(combined_train['MarkDown3'],combined_train['Weekly_Sales']);
combined_train["day"] = [t.dayofweek for t in pd.DatetimeIndex(combined_train.Date)]

combined_train["month"] = [t.month for t in pd.DatetimeIndex(combined_train.Date)]

combined_train['year'] = [t.year for t in pd.DatetimeIndex(combined_train.Date)]
combined_train.head()
combined_train.drop('Date',axis=1,inplace=True)
combined_train.IsHoliday = combined_train.IsHoliday.astype(int)
combined_train.head()
combined_train= pd.get_dummies(combined_train,drop_first=True)
combined_train.head()
from sklearn.feature_selection import f_regression,mutual_info_regression,SelectKBest

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

sc = StandardScaler()
X = combined_train.drop('Weekly_Sales',axis=1)

y = combined_train['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
sel_f = SelectKBest(f_regression,k=15).fit(X_train,y_train)
X.columns[sel_f.get_support()]
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

mse = []

for col in X_train.columns:

  reg = DecisionTreeRegressor()

  reg.fit(X_train[col].to_frame(),y_train)

  y_pred = reg.predict(X_test[col].to_frame())

  mse.append(mean_squared_error(y_test,y_pred))



mse = pd.DataFrame(data=mse,columns=['roc_auc_score'])

mse.index = X_train.columns
sns.set(rc={'figure.figsize':(20,8)})

mse.sort_values(by='roc_auc_score').plot.bar();
sns.heatmap(combined_train.corr(),linewidths=0.5,annot=True);
X_new = combined_train.drop(['day','year','Weekly_Sales','MarkDown1','MarkDown3'],axis=1)

y_new = combined_train['Weekly_Sales']
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error

rf_new = RandomForestRegressor(n_estimators=50,criterion='mse')

model_rf_new = rf_new.fit(X_train_new,y_train_new)

y_pred_rf_new = model_rf_new.predict(X_test_new)

print(np.sqrt(mean_squared_error(y_test_new,y_pred_rf_new)))
y_pred_df = pd.DataFrame(data=y_pred_rf_new,columns=['Prediction'])

n = X_test_new['month']

n.reset_index(drop=True,inplace=True)

y_pred_df['month'] = n
sns.lineplot(x=y_pred_df['month'],y=y_pred_df['Prediction']);