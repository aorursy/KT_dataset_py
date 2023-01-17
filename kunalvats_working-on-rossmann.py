# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import cufflinks as cf
import tensorflow as tf
import pickle
#from fastai.structured import *
#from fastai.column_data import *
from IPython.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()
init_notebook_mode(connected=True)
%matplotlib inline
Path = "../input/"
tables = ['train','store']
table = [pd.read_csv(f'{Path}{fname}.csv',low_memory=False) for fname in tables]
train, store = table
for t in table:
    display(t.head())
# for t in table:
#     display(DataFrameSummary(t).summary())
print(len(train))
len(store)
# Open
fig, (axis1) = plt.subplots(1,1,figsize=(15,8))
sns.countplot(x='Open',hue='DayOfWeek', data=train,palette="husl", ax=axis1)


# Drop Open column
# train.drop("Open", axis=1, inplace=True)
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

train_store['Date'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.date())
train_store['Year'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.year)
train_store['Month'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.month)
train_store['Day'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.day)
c = '#386B7F' #Basic RGB
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
              ) 
# sales trends
sns.catplot(data = train_store, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = c,
           kind = 'boxen') 
# Compute the correlation matrix 
# exclude 'Open' variable
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()
# sale per customer trends
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo2',
               palette = 'RdPu') 

train_store.head(2)
train_store[:10000].iplot(kind='scatter',y='Sales',x='Customers',mode='markers',size=10,
                          xTitle='Number of Customers', yTitle='Sales')
# train_store[:10000].iplot(kind='scatter3d',y='Sales',x='Customers',
#                           z='CompetitionDistance',mode='markers',size=10 )
# SchoolHoliday

# Plot
sns.countplot(x='SchoolHoliday', data=train)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='SchoolHoliday', y='Sales', data=train, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Customers', data=train, ax=axis2)
train_store[:1050].dropna().iplot(kind='spread',x='Date',y=['Customers','Sales']
                                  ,xTitle='Date',yTitle='Sales')
train.iplot()
train_store[:1050].dropna().iplot(kind='spread',x='Date',y='Sales',
                                  xTitle='Date',yTitle='Sales',dash='dashdot',theme='white')
#train_store.iplot(kind='histogram',x)
merged = train_store
#Working with missing data

imp = Imputer(missing_values='NaN',strategy='mean')
imp.fit(merged['CompetitionDistance'].values.reshape(-1,1))
merged['CompetitionDistance'] = imp.transform(merged['CompetitionDistance'].values.reshape(-1,1))

imp1 = Imputer(strategy='median')
imp1.fit(merged['CompetitionOpenSinceYear'].values.reshape(-1,1))
merged['CompetitionOpenSinceYear'] = imp.transform(merged['CompetitionOpenSinceYear'].values.reshape(-1,1))

imp1.fit(merged['CompetitionOpenSinceMonth'].values.reshape(-1,1))
merged['CompetitionOpenSinceMonth'] = imp.transform(merged['CompetitionOpenSinceMonth'].values.reshape(-1,1))
#Dropping columns with excessive null values
merged = merged.drop(['Promo2SinceWeek','Promo2SinceYear','PromoInterval'],axis=1)
merged = pd.concat([merged,pd.get_dummies(merged['StateHoliday'],prefix='StateHoliday',drop_first=True)],axis=1)
merged = pd.concat([merged,pd.get_dummies(merged['StoreType'],prefix='StoreType',drop_first=True)],axis=1)
merged = pd.concat([merged,pd.get_dummies(merged['Assortment'],prefix='Assortment',drop_first=True)],axis=1)
merged.drop(['StateHoliday','StoreType','Assortment','Date'],axis=1,inplace=True)
merged.head()
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lr = LinearRegression()
rf = RandomForestRegressor()
merged.columns
X = ['Store', 'DayOfWeek', 'Open', 'Promo',
       'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Year', 'Month', 'Day',
       'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'StoreType_b',
       'StoreType_c', 'StoreType_d', 'Assortment_b', 'Assortment_c']
y = ['Sales']
x_train, x_eval, y_train, y_eval = train_test_split(merged[X], merged[y],
                                                    test_size=0.3, random_state=101)
print(x_train.shape)
print(y_train.shape)

print(x_eval.shape)
print(y_eval.shape)
lr.fit(x_train,y_train)
prediction = lr.predict(x_eval)
from sklearn.metrics import r2_score, mean_squared_error,explained_variance_score
print(r2_score(y_eval, prediction))
print(mean_squared_error(y_eval, prediction))
print(explained_variance_score(y_eval, prediction))

rf = RandomForestRegressor()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_eval)
print(r2_score(y_eval, rf_pred))
print(mean_squared_error(y_eval, rf_pred))
print(explained_variance_score(y_eval, rf_pred))

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_eval)
print(r2_score(y_eval, knn_pred))
print(mean_squared_error(y_eval, knn_pred))
print(explained_variance_score(y_eval, knn_pred))
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_eval)
print(r2_score(y_eval, dt_pred))
print(mean_squared_error(y_eval, dt_pred))
print(explained_variance_score(y_eval, dt_pred))
# Will take so much time
# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor()
#mlp.fit(x_train,y_train)
# mlp_pred = mlp.predict(x_eval)
# print(r2_score(y_eval, mlp_pred))
# print(mean_squared_error(y_eval, mlp_pred))
# print(explained_variance_score(y_eval, mlp_pred))
