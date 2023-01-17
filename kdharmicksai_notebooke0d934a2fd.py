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
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error 
train_ds=pd.read_csv("/kaggle/input/wecrec2020/Train_data.csv")
test_ds=pd.read_csv("/kaggle/input/wecrec2020/Test_data.csv")
train_ds.head(10)
test_ds.head(10)
train_ds.describe()
train_ds.drop(['Unnamed: 0','F1','F2'],axis=1,inplace=True)
test_index=test_ds['Unnamed: 0']
test_ds.drop(['Unnamed: 0','F1','F2'],axis=1,inplace=True)
print(test_ds.head(10))
print(train_ds.head(10))
x_train = train_ds.drop(['O/P'],axis=1)
y_train = train_ds['O/P']
train_x,test_x,train_y,test_y = train_test_split(x_train,y_train, test_size=0.0001, random_state=50)
XGBModel = XGBRegressor(max_depth=7,learning_rate=0.2,colsample_bylevel=0.9)
XGBModel.fit(train_x,train_y ,verbose =True )


XGBpredictions = XGBModel.predict(test_x)

MAE = mean_squared_error(test_y , XGBpredictions)
print('XGBoost validation MAE = ',np.sqrt(MAE))
XGBpredictions = XGBModel.predict(train_x)

MAE = mean_squared_error(train_y , XGBpredictions)
print('XGBoost validation MAE = ',np.sqrt(MAE))
pred=XGBModel.predict(test_ds)
pred
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output10.csv', index=False)
import matplotlib.pyplot as plt
corrmat = train_ds.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(train_ds[top_corr_features].corr(),annot=True,cmap="RdYlGn")
x_ds = train_ds.drop(['F6','F10','F13','F14','O/P'],axis=1)
x_test=test_ds.drop(['F6','F10','F13','F14'],axis=1)
y_ds = train_ds['O/P']
x_ds
x_test
train_x,test_x,train_y,test_y = train_test_split(x_ds,y_ds, test_size=0.0001, random_state=80)
XGBModel = XGBRegressor()
XGBModel.fit(train_x,train_y ,verbose =True )


XGBpredictions = XGBModel.predict(test_x)
MSE = mean_squared_error(test_y , XGBpredictions)
print('XGBoost validation RMSE = ',np.sqrt(MSE))

XGBpredictions = XGBModel.predict(train_x)
MSE = mean_squared_error(train_y , XGBpredictions)
print('XGBoost train RMSE = ',np.sqrt(MSE))
XGBModel
pred=XGBModel.predict(x_test)
pred
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output12.csv', index=False)