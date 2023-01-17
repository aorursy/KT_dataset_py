# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sales = pd.read_csv("../input/train.csv")
sales.head()
stores = pd.read_csv("../input/store.csv")
stores.head()
sales = sales.sample(frac= 1)

data = pd.merge(sales, stores, 'left', 'Store')
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Day'] = data['Date'].dt.day
data['StateHoliday'] = data['StateHoliday'].replace(0, '0')
data.head()
y = "Sales"
g = sns.distplot(data[y])
g.set_title("Data Distribution")

zero_sales = data[data[y]==0].copy()
data_nz =  data[data[y]!=0].copy()
fig, ax = plt.subplots (1,4, figsize=(20,4))
sns.barplot(['Size'], [len(zero_sales)], ax=ax[0])
sns.countplot('DayOfWeek', data=zero_sales, ax=ax[1])
sns.countplot('Open', data=zero_sales, ax=ax[2])
sns.countplot('Promo', data=zero_sales, ax=ax[3])
plt.tight_layout()
zero_sales_promo = data[(data[y]==0) & (data['Promo'])& (data['Open'])].copy()
zero_sales_promo
fig, ax = plt.subplots (1,4, figsize=(20,4))
sns.pointplot('DayOfWeek', 'Sales', data=data_nz, estimator=np.sum, ax=ax[0])
sns.pointplot('Day', 'Sales', data=data_nz, estimator=np.sum, ax=ax[1])
sns.pointplot('Month', 'Sales', data=data_nz, estimator=np.sum, ax=ax[2])
sns.pointplot('Year', 'Sales', data=data_nz, estimator=np.sum, ax=ax[3])
plt.tight_layout()

fig, ax = plt.subplots (1,5, figsize=(25,4))
sns.boxplot('Promo', 'Sales', data=data_nz, ax=ax[0])
sns.boxplot('StoreType', 'Sales', data=data_nz, ax=ax[1])
sns.boxplot('Assortment', 'Sales', data=data_nz, ax=ax[2])
sns.boxplot('StoreType', 'Sales', 'Assortment', data=data_nz, ax=ax[3])
sns.boxplot('StoreType', 'Sales', 'Promo', data=data_nz, ax=ax[4])
plt.tight_layout()
grid = sns.FacetGrid(data_nz, col="StoreType", row="Promo", palette="tab10", col_order="abcd")
grid.map(sns.pointplot, "Month", "Sales")
data_nz['SalesPerCustomer'] = data_nz[y]/data_nz['Customers']
grid = sns.FacetGrid(data_nz, col="StoreType", row="Promo", palette="tab10", col_order="abcd")
grid.map(sns.pointplot, "Month", "SalesPerCustomer")
fig, ax = plt.subplots (1,2, figsize=(25,4))
sns.boxplot('StateHoliday', 'Sales', data=data_nz, ax=ax[0])
sns.boxplot('StateHoliday', 'SalesPerCustomer', data=data_nz, ax=ax[1])
plt.tight_layout()
fig, ax = plt.subplots (1,4, figsize=(25,4))
sns.boxplot('Month', 'Sales', data=data_nz[data_nz['StateHoliday']=='a'], ax=ax[0])
sns.boxplot('Month', 'Sales', data=data_nz[data_nz['StateHoliday']=='b'], ax=ax[1])
sns.boxplot('Month', 'Sales', data=data_nz[data_nz['StateHoliday']=='c'], ax=ax[2])
sns.boxplot('Month', 'Sales', data=data_nz[data_nz['StateHoliday']=='0'], ax=ax[3])
plt.tight_layout()
g = sns.distplot(data_nz[data_nz['CompetitionDistance']<1600]['CompetitionDistance'].dropna())

data_nz.loc[data_nz['CompetitionDistance']>2000, 'CompetitionDistance']= 2000
g = sns.jointplot("CompetitionDistance", "Sales", data=data_nz.sample(frac=.2), kind="reg", color="m", height=7)
plt.figure(figsize=(15,6))
mask = np.zeros_like(data_nz.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data_nz.corr(), cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
 
data_nz.head()
data_nz.loc[pd.isnull(data_nz['PromoInterval']), 'PromoInterval']= 'None'
cols = ['Date', 'Month', 'Day', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'CompetitionDistance', 'Sales']
import numpy as np
from sklearn.model_selection import cross_val_predict

tmp= pd.pivot_table(data_nz, ['Date'], "Store", aggfunc="count").reset_index().sort_values('Date', ascending=False).head(100)
top_stores = tmp["Store"].values
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

params = {"objective": "reg:linear", # for linear regression
          "booster" : "gbtree",   # use tree based models 
          "eta": 0.1,   # learning rate
          "max_depth": 10,    # maximum depth of a tree
          "subsample": 0.8,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
          "silent": 1,   # silent mode
          "seed": 10   # Random number seed
          }
num_boost_round = 1000
cols = ['Date', 'Store', 'Month', 'Day', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'CompetitionDistance', 'Sales']
test = pd.read_csv('../input/test.csv')
test = pd.merge(test, stores, 'left', 'Store')
test.head()
test['Date'] = pd.to_datetime(test['Date'])
test['Day'] = test['Date'].dt.day
test['Month'] = test['Date'].dt.month
test.loc[pd.isnull(test['PromoInterval']), 'PromoInterval']= 'None'
test.head()
cols = ['Store',
 'Month',
 'Day',
 'DayOfWeek',
 'Promo',
 'StateHoliday',
 'StoreType',
 'CompetitionDistance',
 'Sales']
from sklearn.ensemble import RandomForestRegressor

result = pd.Series()
ids = test['Store'].unique()
dist = np.median(data_nz[pd.notna(data_nz['CompetitionDistance'])]['CompetitionDistance'])
#for i in ids:  
X_train = data_nz#[data_nz['Store']== i].copy()
X_train['Store'] =X_train['Store'].astype('object')
X_train= X_train[cols]
X_test  = test#[test['Store']== i]
X_test['Store'] =X_test['Store'].astype('object')
holidays_test = X_test['StateHoliday'].unique().tolist()
holidays_train = X_train['StateHoliday'].unique().tolist()
holidays = list(set(holidays_test) & set(holidays_train))
storeType = X_train['StoreType'].unique().tolist()

X_train = X_train[(X_train['StateHoliday'].isin(holidays))]
X_test = X_test[(X_test['StateHoliday'].isin(holidays))]
store_ind = X_test["Id"]
X_test = X_test[cols[:-1]].copy() 
Y_train = np.log(X_train["Sales"])
#X_test = X_test[(X_test['StoreType'].isin(storeType))]
X_train.drop(["Sales"], axis=1,inplace=True)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())
X_train['CompetitionDistance'] = X_train['CompetitionDistance'].fillna(dist)
X_test['CompetitionDistance'] = X_test['CompetitionDistance'].fillna(dist)
X_train = pd.get_dummies(X_train).values
X_test = pd.get_dummies(X_test).values

#dtrain = xgb.DMatrix(X_train, Y_train)
#watchlist = [(dtrain, 'train')]
#estimator = xgb.train(params, dtrain, num_boost_round)

estimator = RandomForestRegressor(n_estimators=200)
estimator.fit(X_train, Y_train)
#Y_pred = estimator.predict(xgb.DMatrix(X_test))
Y_pred = estimator.predict(X_test)
result = result.append(pd.Series(np.exp(Y_pred), index=store_ind))
result = pd.DataFrame({ "Id": result.index, "Sales": result.values})
merged_test = pd.merge(test, result, 'left', ['Id']) 
merged_test.loc[ merged_test.Open == 0, 'Sales' ] = 0 
sub = merged_test[['Id', y]].copy() 
sub[y] = sub[y].fillna(0) 
sub.to_csv('submission.csv', index=False) 

sub.info()