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
df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_weather = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')


for i in range(len(df)):
    A = df['DATE_TIME'][i].split(' ')[0].split('-')
    df['DATE_TIME'][i] = '-'.join(A[-1::-1])
for i in range(len(df_weather)):
    df_weather['DATE_TIME'][i] = df_weather['DATE_TIME'][i].split(' ')[0]
#A = df['DATE_TIME'][0].split(' ')[0].split('-')
df = df.drop(['PLANT_ID','SOURCE_KEY'],axis=1)
df_weather = df_weather.drop(['PLANT_ID','SOURCE_KEY'],axis=1)
df_trans = df.groupby(['DATE_TIME']).mean().reset_index()
df_trans_2 = df_weather.groupby(['DATE_TIME']).mean().reset_index()
df_big = pd.concat([df_trans,df_trans_2.drop('DATE_TIME',axis=1)],axis=1)
Alltime = df_big['DATE_TIME'].tolist()
Alltime
def Generate_trainset(df_temp,day):
    A = []
    for i in range(2,len(df_temp)-day):
        df1 = df_temp[ df_temp['DATE_TIME']==Alltime[i-2] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_1','AC_POWER':'AC_POWER_1','DAILY_YIELD':'DAILY_YIELD_1','TOTAL_YIELD':'TOTAL_YIELD_1','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_1','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_1','IRRADIATION':'IRRADIATION_1'}).set_index([pd.Series([i-2])])
        df2 = df_temp[ df_temp['DATE_TIME']==Alltime[i-1] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_2','AC_POWER':'AC_POWER_2','DAILY_YIELD':'DAILY_YIELD_2','TOTAL_YIELD':'TOTAL_YIELD_2','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_2','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_2','IRRADIATION':'IRRADIATION_2'}).set_index([pd.Series([i-2])])
        df3 = df_temp[ df_temp['DATE_TIME']==Alltime[ i ] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_3','AC_POWER':'AC_POWER_3','DAILY_YIELD':'DAILY_YIELD_3','TOTAL_YIELD':'TOTAL_YIELD_3','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_3','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_3','IRRADIATION':'IRRADIATION_3'}).set_index([pd.Series([i-2])])
        dfAns = df_temp[ df_temp['DATE_TIME']==Alltime[i+day] ].copy().set_index([pd.Series([i-2])])['TOTAL_YIELD']
        #if i==3:
        #    print(df1,df2,df3,dfAns,pd.concat([df1,df2,df3,dfAns],axis=1))
        A+= [pd.concat([df1,df2,df3,dfAns],axis=1)]
    return pd.concat(A)
data_3day = Generate_trainset(df_big,3)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math

def Kfold(I,day,df1,df2):
    df = df1.loc[I::10].reset_index().drop('index',axis=1)
    df_weather = df2.loc[I::10].reset_index().drop('index',axis=1)
    
    for i in range(len(df)):
        A = df['DATE_TIME'][i].split(' ')[0].split('-')
        df['DATE_TIME'][i] = '-'.join(A[-1::-1])
    for i in range(len(df_weather)):
        df_weather['DATE_TIME'][i] = df_weather['DATE_TIME'][i].split(' ')[0]
        
    df = df.drop(['PLANT_ID','SOURCE_KEY'],axis=1)
    df_weather = df_weather.drop(['PLANT_ID','SOURCE_KEY'],axis=1)
    df_trans = df.groupby(['DATE_TIME']).mean().reset_index()
    df_trans_2 = df_weather.groupby(['DATE_TIME']).mean().reset_index()
    
    df_big = pd.concat([df_trans,df_trans_2.drop('DATE_TIME',axis=1)],axis=1)
    
    Alltime = df_big['DATE_TIME'].tolist()
    df_temp = df_big
    A = []
    for i in range(2,len(df_temp)-day):
        df1 = df_temp[ df_temp['DATE_TIME']==Alltime[i-2] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_1','AC_POWER':'AC_POWER_1','DAILY_YIELD':'DAILY_YIELD_1','TOTAL_YIELD':'TOTAL_YIELD_1','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_1','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_1','IRRADIATION':'IRRADIATION_1'}).set_index([pd.Series([i-2])])
        df2 = df_temp[ df_temp['DATE_TIME']==Alltime[i-1] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_2','AC_POWER':'AC_POWER_2','DAILY_YIELD':'DAILY_YIELD_2','TOTAL_YIELD':'TOTAL_YIELD_2','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_2','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_2','IRRADIATION':'IRRADIATION_2'}).set_index([pd.Series([i-2])])
        df3 = df_temp[ df_temp['DATE_TIME']==Alltime[ i ] ].copy().drop('DATE_TIME',axis=1).rename(columns={'DC_POWER':'DC_POWER_3','AC_POWER':'AC_POWER_3','DAILY_YIELD':'DAILY_YIELD_3','TOTAL_YIELD':'TOTAL_YIELD_3','AMBIENT_TEMPERATURE':'AMBIENT_TEMPERATURE_3','MODULE_TEMPERATURE':'MODULE_TEMPERATURE_3','IRRADIATION':'IRRADIATION_3'}).set_index([pd.Series([i-2])])
        dfAns = df_temp[ df_temp['DATE_TIME']==Alltime[i+day] ].copy().set_index([pd.Series([i-2])])['TOTAL_YIELD']
        #if i==3:
        #    print(df1,df2,df3,dfAns,pd.concat([df1,df2,df3,dfAns],axis=1))
        A+= [pd.concat([df1,df2,df3,dfAns],axis=1)]
    dataset = pd.concat(A)
    X = dataset.drop('TOTAL_YIELD',axis=1)
    y = dataset['TOTAL_YIELD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    Predict = reg.predict(X_test)
    y_test_list = y_test.tolist()
    N = len(y_test_list)
    MSE = sum([ ((y_test_list[e]-Predict[e])*(y_test_list[e]-Predict[e])) for e in range(N)])/N
    RMSE = math.sqrt(MSE)
    #print('Fold '+str(I+1)+' =',RMSE)
    return 'Fold '+str(I+1)+' RMSE = '+str(RMSE)
df1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
report = 'TOTAL_YIELD of 3days for predicts next 3 day\n'
for i in range(10):
    report += Kfold(i,3,df1,df2) +'\n'
report += '\n\nTOTAL_YIELD of 3 days for predicts next 7 day\n'
for i in range(10):
    report += Kfold(i,7,df1,df2) + '\n'
print(report)
