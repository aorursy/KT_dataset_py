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
import matplotlib # 시각화하는 패키지  
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
train
train['datetime'] = pd.to_datetime(train['datetime'])
train['Year'] = train['datetime'].dt.year

train.info()
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
train['datetime']=pd.to_datetime(train['datetime'])
train['Year'] = train['datetime'].dt.year
train['Month'] = train['datetime'].dt.month
train['Day'] = train['datetime'].dt.day
train['WDay'] = train['datetime'].dt.weekday
train['Hour'] = train['datetime'].dt.hour
train.head()
train['temp'].hist()
# 날씨는 10도 단위로 그룹핑 
for i in train.index:
    t = train.loc[i,'temp']
    if (t >= 0) and (t < 10):
        train.loc[i,'tempGroup'] = '0~9'
    elif (t >= 10) and (t < 20):
        train.loc[i,'tempGroup'] = '10~19'
    elif t >= 20 and t < 30:
        train.loc[i,'tempGroup'] = '20~29'
    elif t >= 30 and t < 40:
        train.loc[i,'tempGroup'] = '30~39'
    elif t >= 40 :
        train.loc[i,'tempGroup'] = '40~'
    else:
        train.loc[i,'tempGroup'] = 'NoValue'
train
train['atemp'].hist()
# 날씨는 10도 단위로 그룹핑 
for i in train.index:
    t = train.loc[i,'atemp']
    if (t >= 0) and (t < 10):
        train.loc[i,'atempGroup'] = '0~9'
    elif (t >= 10) and (t < 20):
        train.loc[i,'atempGroup'] = '10~19'
    elif t >= 20 and t < 30:
        train.loc[i,'atempGroup'] = '20~29'
    elif t >= 30 and t < 40:
        train.loc[i,'atempGroup'] = '30~39'
    elif t >= 40 :
        train.loc[i,'atempGroup'] = '40~'
    else:
        train.loc[i,'atempGroup'] = 'NoValue'
train['humidity'].hist()
# 습도는 20 단위로 그룹핑 
for i in train.index:
    t = train.loc[i,'humidity']
    if t >= 0 and t <= 19:
        train.loc[i,'humidityGroup'] = '0~19'
    elif t >= 20 and t <= 39:
        train.loc[i,'humidityGroup'] = '20~39'
    elif t >= 40 and t <= 59:
        train.loc[i,'humidityGroup'] = '40~59'
    elif t >= 60 and t <= 79:
        train.loc[i,'humidityGroup'] = '60~79'
    elif t >= 80 :
        train.loc[i,'humidityGroup'] = '80~'
    else:
        train.loc[i,'humidityGroup'] = 'NoValue'
train['windspeed'].hist() 
# 풍속은 10 단위로 그룹핑 
for i in train.index:
    t = train.loc[i,'windspeed']
    if t >= 0 and t < 10:
        train.loc[i,'windspeedGroup'] = '0~9'
    elif t >= 10 and t < 20:
        train.loc[i,'windspeedGroup'] = '10~19'
    elif t >= 20 and t < 30:
        train.loc[i,'windspeedGroup'] = '20~29'
    elif t >= 30 :
        train.loc[i,'windspeedGroup'] = '30~'
    else:
        train.loc[i,'windspeedGroup'] = 'NoValue'
train.head()
round(train.groupby(['tempGroup']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['atempGroup']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['humidityGroup']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['windspeedGroup']).agg({'count':np.sum})/10000,1).plot(kind='barh')
# idx = (train['windspeedGroup'] == 'NoValue')
# train.loc[idx,:]
round(train.groupby(['holiday','workingday']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['WDay','holiday']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['WDay','workingday']).agg({'count':np.sum})/10000,1).plot(kind='barh')
round(train.groupby(['weather']).agg({'count':np.sum})/10000,1).plot(kind='barh')
# pd.melt(plot_data.reset_index(),id_vars=['Month'])
plot_data = train.groupby(['Year','Month']).\
agg({'count':np.sum}).\
pivot_table(index='Month',columns='Year')['count']
plot_data
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)

ax.legend()
ax.set_xticks(np.arange(1,12,3))
ax.set_yticks(np.arange(20000,140000,10000))
ax.grid(axis='both',alpha=0.3)

# plt.show()

plot_data = train.groupby(['Month','Hour']).agg({'count':np.sum}).\
pivot_table(index='Hour',columns='Month')['count']
plot_data
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
ax.set_xticks(x)
ax.grid(axis='both',alpha=0.3)
ax.legend()
plt.show()
train2 = train[['Year','Month','Hour','weather','temp','atemp','humidity','windspeed']].copy() 
train2
train2 = train[['Year','Month','Hour','weather','temp','atemp','humidity','windspeed']].copy() 
X = train2.copy()
y = train['count'].copy()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3)

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = linear_model.LinearRegression()
model = RandomForestRegressor()
model.fit(X_train, y_train)
# X_vaild와 y_vaild로 예측을 연습 
# model.predict(X_valid)
print(model.score(X_valid,y_valid))
print(model.predict(X_valid))

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
test['datetime']=pd.to_datetime(test['datetime'])
test['Year'] = test['datetime'].dt.year
test['Month'] = test['datetime'].dt.month
test['Hour'] = test['datetime'].dt.hour
test = test[['Year','Month','Hour','weather','temp','atemp','humidity','windspeed']].copy()
result = model.predict(test)
print(result)
submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
# submission
submission['count'] = result

submission['datetime']=pd.to_datetime(submission['datetime'])
submission['Year'] = submission['datetime'].dt.year
submission['Month'] = submission['datetime'].dt.month
submission['Hour'] = submission['datetime'].dt.hour

plot_data = submission.groupby(['Year','Month']).agg({'count':np.sum}).\
pivot_table(index='Month',columns='Year')['count']
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
ax.set_xticks(x)
ax.grid(axis='both',alpha=0.3)
ax.legend()
plt.show()
plot_data = submission.groupby(['Month','Hour']).agg({'count':np.sum}).\
pivot_table(index='Hour',columns='Month')['count']
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
ax.set_xticks(x)
ax.grid(axis='both',alpha=0.3)
ax.legend()
plt.show()
