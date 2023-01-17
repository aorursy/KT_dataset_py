import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv',dtype={'StateHoliday':object},parse_dates=[2])
test = pd.read_csv('../input/test.csv',dtype={'StateHoliday':object},parse_dates=[3])
stores = pd.read_csv('../input/store.csv')
train.head()
stores.head()
#Finding the number of NAs in each column of train data:
train.apply(lambda x: sum(x.isnull()))
#Finding the number of NAs in each column of test data:
test.apply(lambda x: sum(x.isnull()))
train.describe()
train['Sales'].hist(bins=60)
train[train['Sales']==0].shape[0]
train[train['Open']==0].shape[0]
#Assume store is open if it is null in TEST
test.fillna(1, inplace=True)
#Consider only stores that are open
train = train[train["Open"] != 0]
#Consider only rows that have sales greater than zero.
train = train[train["Sales"] > 0]
#Applying log transformation on the Sales attribute
train['log_sales'] = np.log(train['Sales'])
#Plotting the histogram of Log Sales
train['log_sales'].hist(bins=60)
#Merging the train and test datasets with 'Store' dataset
train = pd.merge(train, stores, on='Store')
test = pd.merge(test, stores, on='Store')
#Replacing NA values from Store dataset with 0 (if any)
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
#Let us define a function 'Coding' which encodes a column by taking a column & encoding rules as parameter
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded
#Encoding StateHoliday as 1 or 0
train["StateHoliday"] = coding(train["StateHoliday"], {'a':1, 'b':1, 'c':1})
test["StateHoliday"] = coding(test["StateHoliday"], {'a':1, 'b':1, 'c':1})
#Converting the StateHoliday values into Float
train['StateHoliday'] = train['StateHoliday'].astype(float)
test['StateHoliday'] = test['StateHoliday'].astype(float)
#Converting the SchoolHoliday values into Float
train['SchoolHoliday'] = train['SchoolHoliday'].astype(float)
test['SchoolHoliday'] = test['SchoolHoliday'].astype(float)
train['Year'] = train.Date.dt.year
test['Year'] = test.Date.dt.year
train['Month'] = train.Date.dt.month
test['Month'] = test.Date.dt.month
train['Day'] = train.Date.dt.day
test['Day'] = test.Date.dt.day
sns.regplot(x='Day',y='Sales',data=train)
sns.regplot(x='Month',y='Sales',data=train)
sns.regplot(x='Year',y='Sales',data=train)
train = pd.get_dummies(train,columns=['StoreType','Assortment','Year'])
test = pd.get_dummies(test,columns=['StoreType','Assortment','Year'])
test['Year_2013']=0
test['Year_2014']=0
X = train[train.columns.difference(['Sales','log_sales','Store','Date','Customers','CompetitionOpenSinceYear','Promo2SinceYear','PromoInterval'])]
y = train['log_sales']

#Splitting the train data for training and validation 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)

#Removing unwanted columns in Test data
X_test = test[test.columns.difference(['Id','Store','Date','Customers','CompetitionOpenSinceYear','Promo2SinceYear','PromoInterval'])]
lm = LinearRegression()

lm.fit(X_train,y_train)

lm_pred = lm.predict(X_val)
rmse_lm = np.sqrt(mean_squared_error(y_val,lm_pred))
rmse_lm
rf = RandomForestRegressor(n_estimators=100,max_depth=15)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val,rf_pred))
rmse_rf
xgb = XGBRegressor(max_depth=15,n_jobs=4,n_estimators=120,subsample=0.7)

xgb.fit(X_train,y_train)

xg_pred = xgb.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val,xg_pred))
rmse_xgb
ranking = np.argsort(-xgb.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=xgb.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()
xgb_final = XGBRegressor(max_depth=15,n_jobs=4,n_estimators=120,subsample=0.7)

xgb_final.fit(X,y)

xgb_pred_final = xgb.predict(X_test)
xgb_sub = pd.DataFrame({
    'Id':test['Id'].astype(int),
    'Sales': np.exp(xgb_pred_final)
},columns=['Id','Sales'])

xgb_sub_final = xgb_sub.sort_values(by='Id',ascending=True)

xgb_sub_final.to_csv('xgb_sub_final.csv',index=False)
