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
import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns 

from operator import itemgetter

plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus']=False

%matplotlib inline

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor



from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb

import xgboost as xgb



from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error



import os

print(os.listdir("../input"))
Train_data=pd.read_excel("../input/machinehackused-cars-sales-price/Data_Train.xlsx")

Test_data=pd.read_excel("../input/machinehackused-cars-sales-price/Data_Test.xlsx")
Train=Train_data.copy().drop(['Price','New_Price'],axis=1)

Test=Test_data.copy().drop('New_Price',axis=1)
data=pd.concat([Train,Test])
data['index']=range(len(data))
data.loc[(data['Power'] == 'null bhp'),'Power']=np.nan
data=data.sort_values(by=['Name'])

data=data.fillna(method='ffill')
data.isnull().sum()
data['Power']=data['Power'].apply(lambda x:x[:-4]).astype('float')
data['Mileage']=data['Mileage'].apply(lambda x:x[:-5]).astype('float')
data['Engine']=data['Engine'].apply(lambda x:x[:-3]).astype('int')
data['Age'] = (2020 - data['Year']).astype(int)

data=data.drop('Year',axis=1)
count_Owner_Type = data['Owner_Type'].unique()

data['Owner_Type'] = data['Owner_Type'].replace(count_Owner_Type, range(len(count_Owner_Type)))



count_Transmission_Type = data['Transmission'].unique()

data['Transmission'] = data['Transmission'].replace(count_Transmission_Type, range(len(count_Transmission_Type)))



count_Fuel_Type = data['Fuel_Type'].unique()

data['Fuel_Type'] = data['Fuel_Type'].replace(count_Fuel_Type, range(len(count_Fuel_Type)))



count_Location = data['Location'].unique()

data['Location'] = data['Location'].replace(count_Location, range(len(count_Location)))



count_Name = data['Name'].unique()

data['Name'] = data['Name'].replace(count_Name, range(len(count_Name)))
data=data.sort_values(by=['index'])
order = ['Name','Location','Age','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats']

data = data[order]
Train = data.iloc[:6019, :]

Test = data.iloc[6019:, :]
X = Train

Y = Train_data['Price']

X_test = Test
def build_model_xgb(x_train,y_train):

    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\

        colsample_bytree=0.9, max_depth=7)

    model.fit(x_train, y_train)

    return model



def build_model_lgb(x_train,y_train):

    estimator = lgb.LGBMRegressor(num_leaves=100,n_estimators = 150)

    param_grid = {

        'learning_rate': [0.05, 0.1, 0.2, 0.3],

    }

    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(x_train, y_train)

    return gbm

x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=0.3)
model_lgb = build_model_lgb(x_train,y_train)

val_lgb = model_lgb.predict(x_val)

MAE_lgb = mean_absolute_error(y_val,val_lgb)

model_lgb_pre = build_model_lgb(X,Y)

subA_lgb = model_lgb_pre.predict(X_test)
model_xgb = build_model_xgb(x_train,y_train)

val_xgb = model_xgb.predict(x_val)

MAE_xgb = mean_absolute_error(y_val,val_xgb)

model_xgb_pre = build_model_xgb(X,Y)

subA_xgb = model_xgb_pre.predict(X_test)
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb

val_Weighted[val_Weighted<0]=10
sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb
output = pd.DataFrame()



output['price'] = sub_Weighted



output.to_excel('output.xlsx', index = False)
output.head()