# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1_generation = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df1_weather = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df2_generation = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df2_weather = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
print(df1_generation.shape)
print(df1_weather.shape)
print(df1_generation.info())
display(df1_generation.sample(10))
print(df1_weather.info())
display(df1_weather.sample(10))
df1_generation["DATE_TIME"] = pd.to_datetime(df1_generation["DATE_TIME"])
df1_weather["DATE_TIME"] = pd.to_datetime(df1_weather["DATE_TIME"])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df1_generation['NEW_SOURCE_KEY'] = le.fit_transform(df1_generation["SOURCE_KEY"])
df1_generation.sample(10)

df1_generation = df1_generation[['DATE_TIME','DC_POWER','AC_POWER','DAILY_YIELD','TOTAL_YIELD','NEW_SOURCE_KEY']]
df1_weather = df1_weather[['DATE_TIME','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']]
df1 = pd.merge(df1_generation,df1_weather, on="DATE_TIME", how="inner")
df1
df1.info()
df1['DATE_TIME'] = df1['DATE_TIME'].dt.date
df1 = df1.drop('TOTAL_YIELD',axis=1)
df1_sources = df1.groupby(['DATE_TIME','NEW_SOURCE_KEY']).agg({'DC_POWER':'max','AC_POWER':'max','DAILY_YIELD': 'max', 'AMBIENT_TEMPERATURE': 'mean','MODULE_TEMPERATURE':'mean','IRRADIATION':'mean'})
df1_sources.head()
pv = pd.pivot_table(df1_sources, values = 'DAILY_YIELD', index = 'DATE_TIME',
               columns = 'NEW_SOURCE_KEY')
fig,ax1 = plt.subplots(1, 1, figsize=(20,14))
pv.plot(kind='line', ax=ax1)
fig.show()
# 3 column แรกเป็นยอดรวม
sr0 = df1_sources['DC_POWER'].sum(level=0)
sr1 = df1_sources['AC_POWER'].sum(level=0)
sr2 = df1_sources['DAILY_YIELD'].sum(level=0)
# 3 column หลังเป็นค่าเฉลี่ย
sr3 = df1_sources['AMBIENT_TEMPERATURE'].mean(level=0)
sr4 = df1_sources['MODULE_TEMPERATURE'].mean(level=0)
sr5 = df1_sources['IRRADIATION'].mean(level=0)
frame = { 'DC_POWER':sr0,'AC_POWER':sr1,'DAILY_YIELD':sr2,'AMBIENT_TEMPERATURE':sr3,'MODULE_TEMPERATURE':sr4,'IRRADIATION':sr5 } 
  
df_final = pd.DataFrame(frame) 
df_final.sample(5)
fig = plt.figure(figsize=(10,8))
sns.heatmap(df_final.corr(), robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True,cmap='Oranges')
plt.show()
fig = plt.figure(figsize=(18,16))
sns.lineplot(x=df_final.index,y='DAILY_YIELD',data=df_final)
plt.show()
X = df_final.drop('DAILY_YIELD',axis=1)
y = df_final['DAILY_YIELD']
# คำนวณจากค่าเฉลี่ยย้อนหลังตามจำนวนวัน

def x_future(data,days):
    xf = data[-days:,:]
    return [np.mean(xf,axis=0)]
def create_10_fold(X,y):
    for i in range(1,11):
        xtrain = X.drop(X.iloc[i::10,:].index)
        xtest = X.iloc[i::10,:]
        ytrain = y.drop(y.iloc[i::10].index)
        ytest = y.iloc[i::10]
        yield xtrain.values,xtest.values,ytrain.values,ytest.values
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

count = 1
for X_train,X_test,y_train,y_test in create_10_fold(X,y):
    print('='*80)
    print('รอบ ',count)
    print('='*80)
        
    lr = LinearRegression()
    dt = DecisionTreeRegressor()
    mlp = MLPRegressor(max_iter=7000)
    svr = SVR(kernel='linear')
    
    lr_model = lr.fit(X_train,y_train)
    dt_model = dt.fit(X_train,y_train)
    mlp_model = mlp.fit(X_train,y_train)
    svr_model = svr.fit(X_train,y_train)
    
    lr_predicted = lr_model.predict(X_test)
    dt_predicted = dt_model.predict(X_test)
    mlp_predicted = mlp_model.predict(X_test)
    svr_predicted = svr_model.predict(X_test)
    
    lr_rmse = sqrt(mean_squared_error(y_test, lr_predicted))
    dt_rmse = sqrt(mean_squared_error(y_test, dt_predicted))
    mlp_rmse = sqrt(mean_squared_error(y_test, mlp_predicted))
    svr_rmse = sqrt(mean_squared_error(y_test, svr_predicted))
    
    
    X_d3 = x_future(X.values,3)
    X_d4 = x_future(X.values,4)
    X_d5 = x_future(X.values,5)
    X_d9 = x_future(X.values,9)
    
    print('LinearRegression RMSE : ',lr_rmse)
    print('LinearRegression ทำนายผล 3 วันข้างหน้า : ',lr_model.predict(X_d3))
    print('LinearRegression ทำนายผล 4 วันข้างหน้า : ',lr_model.predict(X_d4))
    print('LinearRegression ทำนายผล 5 วันข้างหน้า : ',lr_model.predict(X_d5))
    print('LinearRegression ทำนายผล 9 วันข้างหน้า : ',lr_model.predict(X_d9))
    print('')
    print('DecisionTreeRegressor RMSE : ',dt_rmse)
    print('DecisionTreeRegressor ทำนายผล 3 วันข้างหน้า : ',dt_model.predict(X_d3))
    print('DecisionTreeRegressor ทำนายผล 4 วันข้างหน้า : ',dt_model.predict(X_d4))
    print('DecisionTreeRegressor ทำนายผล 5 วันข้างหน้า : ',dt_model.predict(X_d5))
    print('DecisionTreeRegressor ทำนายผล 9 วันข้างหน้า : ',dt_model.predict(X_d9))
    print('')
    print('MLPRegressor RMSE : ',mlp_rmse)
    print('MLPRegressor ทำนายผล 3 วันข้างหน้า : ',mlp_model.predict(X_d3))
    print('MLPRegressor ทำนายผล 4 วันข้างหน้า : ',mlp_model.predict(X_d4))
    print('MLPRegressor ทำนายผล 5 วันข้างหน้า : ',mlp_model.predict(X_d5))
    print('MLPRegressor ทำนายผล 9 วันข้างหน้า : ',mlp_model.predict(X_d9))
    print('')
    print('SVR RMSE : ',svr_rmse)
    print('SVR ทำนายผล 3 วันข้างหน้า : ',svr_model.predict(X_d3))
    print('SVR ทำนายผล 4 วันข้างหน้า : ',svr_model.predict(X_d4))
    print('SVR ทำนายผล 5 วันข้างหน้า : ',svr_model.predict(X_d5))
    print('SVR ทำนายผล 9 วันข้างหน้า : ',svr_model.predict(X_d9))

    count = count+1


