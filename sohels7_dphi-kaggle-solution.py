# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # to visualise
import seaborn as sns # to visualise
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_train.csv')
data.shape
data.columns
data.head(10)
data.tail(10)
data.info()
data.describe()
o = data.dtypes[data.dtypes == 'object'].index

data[o].describe()
data.nunique()
data.isnull().sum()
plt.figure(figsize=(12, 10))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.tight_layout() 
import missingno as msno 

# Visualize missing values as a matrix 
msno.matrix(data) 
plt.tight_layout() 
# Visualize the number of missing 
# values as a bar chart 
msno.bar(data) 
plt.tight_layout() 
# deleting the neighbourhood_group column as it has only na values in it

data.drop('neighbourhood_group', axis=1, inplace=True)
data.head()
data['name'].isnull().sum()
n = data.loc[pd.isna(data['name']), :].index
data.loc[n]
data['name'].value_counts()
data['name'].mode()
# impute with mode of name

data['name'] = data['name'].fillna(data['name'].mode()[0])
data['last_review'].value_counts()
data['last_review'].mode()
# impute with mode of last_review

data['last_review'] = data['last_review'].fillna(data['last_review'].mode()[0])
data['reviews_per_month'].value_counts()
data['reviews_per_month'].mode()
# impute with mode of reviews_per_month

data['reviews_per_month'] = data['reviews_per_month'].fillna(data['reviews_per_month'].mode()[0])
data['host_name'].value_counts()
data['host_name'].mode()
# impute with mode of host_name

data['host_name'] = data['host_name'].fillna(data['host_name'].mode()[0])
data.isnull().sum()
plt.figure(figsize=(12, 10))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.tight_layout() 
# Target Variable

plt.figure(figsize=(12, 10))
sns.distplot(data['price'])
plt.tight_layout() 
plt.figure(figsize=(12, 10))
sns.boxplot(data['price'])
plt.tight_layout() 
data['price'].describe()
sns.pairplot(data,diag_kind="hist")
plt.tight_layout() 
data.hist(figsize= (16,9))
data['room_type'].value_counts()
plt.figure(figsize=(10, 6))
sns.countplot(x= 'room_type', order= data['room_type'].value_counts().index,data= data,palette='Accent')
plt.figure(figsize=(15, 10))
sns.countplot(x= 'neighbourhood', order= data['neighbourhood'].value_counts().index,data= data,palette='Accent')

plt.xticks(
    rotation=45, 
    horizontalalignment='right'  
)

plt.tight_layout()
plt.figure(figsize=(20, 10))

Filter_price = 800      # as the price is less than 1000 for the most of the houses

sub_data = data[data['price'] < Filter_price]

sns.violinplot(x= 'neighbourhood', y= 'price',data= sub_data,palette='Accent')

plt.xticks(
    rotation=45, 
    horizontalalignment='right'  
)

plt.tight_layout()
plt.figure(figsize=(15,10))
sns.scatterplot(x="longitude", y="latitude", hue="room_type", data=data)                                                                                
plt.tight_layout()
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot = True)
plt.tight_layout()
data.dtypes
data.nunique()
data.head(10)
data.drop(['id','name', 'host_name', 'last_review'],axis=1, inplace=True)
#data['last_review'].value_counts()
'''import datetime

data['last_review'] = pd.to_datetime(data['last_review'])'''
data.head(10)
data = pd.concat([data, pd.get_dummies(data['room_type'], prefix='d')],axis=1)
data.head()
#convert to category dtype

data['neighbourhood'] = data['neighbourhood'].astype('category')
data.dtypes
#use .cat.codes to create new colums with encoded value

data['neighbourhood'] = data['neighbourhood'].cat.codes
data.head()
X = data.drop(['room_type', 'price'],axis=True)

y = data['price']
X.head()
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print('X_train:,y_train:',X_train.shape,y_train.shape)

print('*'*50)

print('X_test:,y_test:',X_test.shape,y_test.shape)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

y_pred = lm.predict(X_test)

y_pred
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_pred))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_pred))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train,y_train)

y_dt = dt.predict(X_test)
y_dt
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_dt))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_dt))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_dt)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_dt))
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=0)
model_rf.fit(X_train,y_train)

y_rf = model_rf.predict(X_test)
y_rf
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, y_rf))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, y_rf))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, y_rf))
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

gb_pred = gbr.predict(X_test)
gb_pred
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, gb_pred))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, gb_pred))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, gb_pred)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, gb_pred))
import xgboost as xgb

xgbr = xgb.XGBRegressor()
xgbr.fit(X_train,y_train)

xgb_pred = xgbr.predict(X_test)
xgb_pred
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, xgb_pred))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, xgb_pred))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, xgb_pred))
from sklearn.model_selection import RandomizedSearchCV

# A parameter grid for XGBoost
params = {
    'colsample_bytree': [0.3,0.5,0.7,1],
    'n_estimators': [300, 500, 800, 1000],
    'learning_rate':[0.01,0.03,0.05,0.07,0.3,0.5,0.1],
    'max_depth': range(2, 20),
    'min_child_weight':[1,3,5,7,10],
    'subsample': [0.5,0.8,1],
}


estimator = xgb.XGBRegressor(random_state=0)

xg_random = RandomizedSearchCV(estimator = estimator, param_distributions = params, cv = 3, n_iter = 100 ,verbose=2, random_state=0,n_jobs = -1,scoring='neg_mean_squared_error')

xg_random.fit(X_train, y_train)
xg_random.best_params_
xg_random.best_estimator_
xg_random.best_score_
xg_rcv = xg_random.best_estimator_.predict(X_test)
xg_rcv
# import metrics library
from sklearn import metrics

# print result of MAE
print('MAE:  ',metrics.mean_absolute_error(y_test, xg_rcv))

# print result of MSE
print('MSE:  ',metrics.mean_squared_error(y_test, xg_rcv))

# print result of RMSE
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, xg_rcv)))

# print result of R^2 Score
print('R^2:  ',metrics.r2_score(y_test, xg_rcv))
data_val = pd.read_csv('/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_validate.csv')
data_val.head(10)
data_val.tail(10)
data_val.isnull().sum()
plt.figure(figsize=(12, 10))
sns.heatmap(data_val.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.tight_layout() 
import missingno as msno 

# Visualize missing values as a matrix 
msno.matrix(data_val) 
plt.tight_layout()
# Visualize the number of missing 
# values as a bar chart 
msno.bar(data_val) 
plt.tight_layout() 
# deleting the neighbourhood_group column as it has only na values in it

data_val.drop('neighbourhood_group', axis=1, inplace=True)
data_val.head()
# impute with mode of name

data_val['name'] = data_val['name'].fillna(data_val['name'].mode()[0])
# impute with mode of last_review

data_val['last_review'] = data_val['last_review'].fillna(data_val['last_review'].mode()[0])
# impute with mode of reviews_per_month

data_val['reviews_per_month'] = data_val['reviews_per_month'].fillna(data_val['reviews_per_month'].mode()[0])
# impute with mode of host_name

data_val['host_name'] = data_val['host_name'].fillna(data_val['host_name'].mode()[0])
plt.figure(figsize=(12, 10))
sns.heatmap(data_val.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.tight_layout() 
data_val.drop(['id', 'name', 'host_name', 'last_review'],axis=1, inplace=True)
data_val.head()
data_val = pd.concat([data_val, pd.get_dummies(data_val['room_type'], prefix='d')],axis=1)
data_val.head()
data_val.dtypes
#convert to category dtype

data_val['neighbourhood'] = data_val['neighbourhood'].astype('category')
data_val.dtypes
#use .cat.codes to create new colums with encoded value

data_val['neighbourhood'] = data_val['neighbourhood'].cat.codes
data_val.head()
data_val.drop('room_type', axis=1, inplace=True)
data_val.head()
xgbpred_val = xgbr.predict(data_val)
xgbpred_val
ind = pd.read_csv('/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_validate.csv')
ind = ind[['id','neighbourhood']]
# To create Dataframe of predicted value with particular respective index
result = pd.DataFrame(xgbpred_val)
result.index = ind['id'] # its important for comparison
result.columns = ["price"]
result.head()
from IPython.display import HTML


result.to_csv('submission1.csv')

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission1.csv')