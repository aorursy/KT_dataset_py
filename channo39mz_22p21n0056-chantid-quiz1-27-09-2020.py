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
import pandas as pd

data = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv") 
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], format="%d-%m-%Y %H:%M")
data2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv") 
data2['DATE_TIME'] = pd.to_datetime(data2['DATE_TIME'], format="%Y-%m-%d %H:%M")
data.describe()
data2.describe()
print(data)
print(data2)
TimeID = pd.merge(data,data2, on=['DATE_TIME','PLANT_ID'])
TimeID.describe()
print(TimeID)
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
def tenper(df,row):
    testdf = df.iloc[[i for i in range(row,df.shape[0],10)]]#ilocตัดมาจากแถวที่กำหนด
    traindf = pd.concat([df,testdf])#ต่อดาต้าหลักกับค่า10%
    traindf = traindf.drop_duplicates(keep=False)
    return traindf,testdf
    
print(TimeID)
TimeID = TimeID.set_index('DATE_TIME') #ทำไห้ DATE_TIME เป็นค่าอ้างอิง(รันได้รอบเดียว)
print(TimeID)
TimeID.describe()
print(TimeID)
print(daily)
train0, test0 = tenper(TimeID, 0)
train1, test1 = tenper(TimeID, 1)
train2, test2 = tenper(TimeID, 2)
train3, test3 = tenper(TimeID, 3)
train4, test4 = tenper(TimeID, 4)
train5, test5 = tenper(TimeID, 5)
train6, test6 = tenper(TimeID, 6)
train7, test7 = tenper(TimeID, 7)
train8, test8 = tenper(TimeID, 8)
train9, test9 = tenper(TimeID, 9)
TrainSet = [train0, train1, train2, train3, train4, train5, train6, train7, train8, train9]
TestSet = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9]
daily = train0[['TOTAL_YIELD','DAILY_YIELD']]
daily.loc[:,'d1'] = (daily.loc[:,'DAILY_YIELD']).shift()
daily.loc[:,'d2'] = (daily.loc[:,'d1']).shift()
daily.loc[:,'d3'] = (daily.loc[:,'d2']).shift()
#daily = daily.dropna()
daily = daily.dropna()
print(daily)
lr = linear_model.LinearRegression()
lr.fit(train0[['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].values,train0['DAILY_YIELD'].values)

yp = lr.predict(train0[['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].values) #1วันข้างหน้า
yt = test0['DAILY_YIELD'].values
from sklearn import metrics, linear_model
print(daily)
lr = linear_model.LinearRegression()
x = daily[['DAILY_YIELD','d1','d2','d3']].values
y = daily['TOTAL_YIELD'].values
lr.fit(x,y)
ly = lr.predict(daily[['DAILY_YIELD','d1','d2','d3']].values)
from sklearn.metrics import mean_squared_error
mean_squared_error(y, ly) #3วัน
daily2 = train1[['TOTAL_YIELD','DAILY_YIELD']]
daily2.loc[:,'d1'] = (daily2.loc[:,'DAILY_YIELD']).shift()
daily2.loc[:,'d2'] = (daily2.loc[:,'d1']).shift()
daily2.loc[:,'d3'] = (daily2.loc[:,'d2']).shift()
daily2.loc[:,'d4'] = (daily2.loc[:,'d3']).shift()
daily2.loc[:,'d5'] = (daily2.loc[:,'d4']).shift()
daily2.loc[:,'d6'] = (daily2.loc[:,'d5']).shift()
daily2.loc[:,'d7'] = (daily2.loc[:,'d6']).shift()


daily2 = daily2.dropna()
lr2 = linear_model.LinearRegression()
x2 = daily2[['DAILY_YIELD','d1','d2','d3','d4','d5','d6','d7']].values
y2 = daily2['TOTAL_YIELD'].values
lr2.fit(x,y)
ly2 = lr.predict(daily2[['DAILY_YIELD','d1','d2','d3','d4','d5','d6','d7']].values)
mean_squared_error(y2, ly2) #7วัน