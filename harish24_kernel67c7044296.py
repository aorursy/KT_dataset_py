# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data=pd.read_csv('/kaggle/input/brent-oil-prices/BrentOilPrices.csv')

oil=data.copy()

oil['Date']=oil['Date'].str.replace(',','',regex=True)

oil['Month']=oil['Date'].str.split(' ').str[0]

oil['date']=oil['Date'].str.split(' ').str[1].astype('int64')

oil['year']=oil['Date'].str.split(' ').str[2].astype('int64')

oil['Month'].replace(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],[1,2,3,4,5,6,7,8,9,10,11,12],inplace=True)

oil=oil.drop(['Date'],axis=1)

corr=oil.corr()

#sns.distplot(oil['Month'],bins=5)

# =============================================================================

# plt.scatter(oil['year'],oil['Price'],c='red')

# plt.xlabel('year')

# plt.ylabel('Price')

# sns.lmplot(x='year',y='Price',data=oil,hue='date')

# =============================================================================

#sns.distplot(oil['Month'],bins=5)

x,y=oil[['year']],oil['Price']

data_DMatrix=xgb.DMatrix(data=x,label=y)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.3,max_depth = 5, alpha = 10, n_estimators = 100)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))

pred_oil=pd.DataFrame({'Predicted':preds})

pred_oil.to_csv('/kaggle/working/pred_oil.csv')