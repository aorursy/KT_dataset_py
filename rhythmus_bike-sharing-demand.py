# 导入库

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()

plt.rcParams['font.family'] = 'STXIHEI'
#读取文件

train = pd.read_csv("../input/bikesharing/train_bike.csv")

train.head()
train.info()
# object转换成datetime格式

train.datetime = pd.to_datetime(train.datetime, format='%Y%m%d %H:%M')  
train['year_month'] = train.datetime.values.astype('datetime64[M]')          # 用numpy处理时间

train.head()
train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train.head()
train['weekday'] = train['datetime'].dt.weekday_name

train.head()
seasonDict = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}

train['season_text'] = train.season.map(seasonDict)

train.head()
weatherDict = {1:'good weaher',2:'cloudy',3:'little rainy or snowy',4:'bad weather'}

train['weather_text'] = train.weather.map(weatherDict)

train.head()
trainCorr = train.corr()

trainCorr
fig1 = plt.figure(figsize=(15,15))

sns.heatmap(trainCorr, annot=True,cmap='YlGnBu',linewidths=.5)

plt.show()
hourSeasonCountSum = train.pivot_table(values='count',

                                 index='hour',

                                 columns='season_text',

                                 aggfunc='sum')



hourSeasonCountSum.head()
hourSeasonCountSum.plot(figsize=(12,5),linestyle='dashed', marker='o')

plt.title('各个季节，24小时内的总租车量情况')

plt.show()
hourSeasonRegSum = train.pivot_table(values='registered',

                                 index='hour',

                                 columns='season_text',

                                 aggfunc='sum')



hourSeasonRegSum.head()
hourSeasonCasSum = train.pivot_table(values='casual',

                                     index='hour',

                                     columns='season_text',

                                     aggfunc='sum')



hourSeasonCasSum.head()
hourSeasonRegSum.plot(figsize=(12,6), linestyle='dashed', marker='o')

plt.title('各个季节，24小时内，注册用户租车量情况')

plt.show()
hourSeasonCasSum.plot(figsize=(12,6) ,linestyle='dashed', marker='o')

plt.title('各个季节，24小时内，非注册用户租车量情况')

plt.show()
hourWeekdaySum = train.pivot_table(values='count',

                                   index='hour',

                                   columns='weekday',

                                   aggfunc='sum')



hourWeekdaySum.head()
hourWeekdaySum.plot.barh(stacked=True, figsize=(15,10))

plt.show()
hourWeekdaySum.plot(figsize=(12,5),linestyle='dashed', marker='o')

plt.title('每周，24小时内的总租车量情况')

plt.show()
hourWeekdayRegSum = train.pivot_table(values='registered',

                                      index='hour',

                                      columns='weekday',

                                      aggfunc='sum')



hourWeekdayRegSum.head()
hourWeekdayCasSum = train.pivot_table(values='casual',

                                      index='hour',

                                      columns='weekday',

                                      aggfunc='sum')



hourWeekdayCasSum.head()
hourWeekdayRegSum.plot(figsize=(12,6) ,linestyle='dashed', marker='o')

plt.title('周，24小时内，注册用户租车量情况')

plt.show()
hourWeekdayCasSum.plot(figsize=(12,6) ,linestyle='dashed', marker='o')

plt.title('周，24小时内，非注册用户租车量情况')

plt.show()
group_month = train.groupby('year_month')

group_month_regSum = group_month.registered.sum()

group_month_regSum.head(2)
group_month_casSum = group_month.casual.sum()

group_month_casSum.head(2)
A = pd.concat([group_month_regSum,group_month_casSum], axis=1)

A.head(2)
A.plot.area(stacked=True, figsize=(12,5), color=['b','g'], alpha=0.8)

plt.legend()

plt.show()
sns.pairplot(train[['temp','atemp','humidity','windspeed','casual','registered']])

plt.show()