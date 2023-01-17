# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#讀取資料

train = pd.read_csv("../input/PM25_train.csv") #load the dataset

test = pd.read_csv("../input/PM25_test.csv")

submission = pd.read_csv('../input/submission.csv')

print(train.shape)

train.head()
#發現有重複值的狀況，因此要移除重複的資料減少資料複雜度

train.drop_duplicates(subset=None, keep='first', inplace=True)

train.reset_index(inplace=True)

train.drop('index',inplace=True,axis=1)

print(train)
#checked missing data

train.info()

test.info()
import seaborn as sns

# correlation calculate

corr = train[['PM10','PM1','Temperature','Humidity',"PM2.5"]].corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r") #center=0, cmap="YlGnBu"
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10,5), sharey=True)

axes[0].boxplot(train['PM10'],showmeans=True)

axes[0].set_title('PM10')



axes[1].boxplot(train['PM1'],showmeans=True)

axes[1].set_title('PM1')



axes[2].boxplot(train['Temperature'],showmeans=True)

axes[2].set_title('Temperature')



axes[3].boxplot(train['Humidity'],showmeans=True)

axes[3].set_title('Humidity')



axes[4].boxplot(train['PM2.5'],showmeans=True)

axes[4].set_title('PM2.5')
# 查看訓練資料的分布狀況(四分位距)

pd.set_option('display.float_format', lambda x: '%.6f' % x)

train.describe()
# 查看測試資料的分布狀況(四分位距)

test.describe()
# 將所有特徵超出三倍標準差的概念將這些Outlier先去掉，避免對Model造成影響

print ("Shape Of The Before Ouliers: ",train.shape)

# train = train[np.abs(train["PM10"]-train["PM10"].mean())<=(3*train["PM10"].std())] 

# train = train[np.abs(train["PM1"]-train["PM1"].mean())<=(3*train["PM1"].std())] 

# train = train[np.abs(train["PM2.5"]-train["PM2.5"].mean())<=(3*train["PM2.5"].std())] 

# train = train[np.abs(train["Temperature"]-train["Temperature"].mean())<=(3*train["Temperature"].std())] 

# train = train[np.abs(train["Humidity"]-train["Humidity"].mean())<=(3*train["Humidity"].std())] 



#IQR = Q3-Q1

IQR = np.percentile(train['PM10'],75) - np.percentile(train['PM10'],25)

#outlier = Q3 + 1.5*IQR 

train=train[train['PM10'] < np.percentile(train['PM10'],75)+1.5*IQR]

#IQR = Q3-Q1

IQR = np.percentile(train['PM1'],75) - np.percentile(train['PM1'],25)

#outlier = Q3 + 1.5*IQR 

train=train[train['PM1'] < np.percentile(train['PM1'],75)+1.5*IQR]

#IQR = Q3-Q1

IQR = np.percentile(train['PM2.5'],75) - np.percentile(train['PM2.5'],25)

#outlier = Q3 + 1.5*IQR 

train=train[train['PM2.5'] < np.percentile(train['PM2.5'],75)+1.5*IQR]

#IQR = Q3-Q1

IQR = np.percentile(train['Temperature'],75) - np.percentile(train['Temperature'],25)

#outlier = Q3 + 1.5*IQR 

train=train[train['Temperature'] < np.percentile(train['Temperature'],75)+1.5*IQR]

#IQR = Q3-Q1

IQR = np.percentile(train['Humidity'],75) - np.percentile(train['Humidity'],25)

#outlier = Q3 + 1.5*IQR 

train=train[train['Humidity'] < np.percentile(train['Humidity'],75)+1.5*IQR]

print ("Shape Of The After Ouliers: ",train.shape)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10,5), sharey=True)

axes[0].boxplot(train['PM10'],showmeans=True)

axes[0].set_title('PM10')



axes[1].boxplot(train['PM1'],showmeans=True)

axes[1].set_title('PM1')



axes[2].boxplot(train['Temperature'],showmeans=True)

axes[2].set_title('Temperature')



axes[3].boxplot(train['Humidity'],showmeans=True)

axes[3].set_title('Humidity')



axes[4].boxplot(train['PM2.5'],showmeans=True)

axes[4].set_title('PM2.5')
print ("Shape Of The PM10 equal 0 from train data: ",train[train['PM10']==0].shape)

print ("Shape Of The PM1 equal 0 from train data: ",train[train['PM1']==0].shape)
data = train.append(test)

data.shape

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)

data['Date'] = pd.to_datetime(data['Date'])

data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')

data['weekday'] = data['Date'].map(lambda x :x.weekday())

data['Hour'] = data['Time'].map(lambda x :x.hour)

data['Minute'] = data['Time'].map(lambda x :x.minute)

data['Year'] = data['Date'].map(lambda x:x.year)

data['Month'] = data['Date'].map(lambda x:x.month)

data['Day'] = data['Time'].map(lambda x:x.day)

data
fig, axes = plt.subplots(nrows=1,ncols=3)

fig.set_size_inches(12, 4)

sns.distplot(data["PM2.5"][0:train.shape[0]],ax=axes[0])

sns.distplot(data["PM10"][0:train.shape[0]],ax=axes[1])

sns.distplot(data["PM1"][0:train.shape[0]],ax=axes[2])



axes[0].set(xlabel='PM2.5',title="distribution of temp")

axes[1].set(xlabel='PM10',title="distribution of atemp")

axes[2].set(xlabel='PM1',title="distribution of humidity")
dataTrain = data[pd.notnull(data['PM2.5'])]

dataTest = data[~pd.notnull(data['PM2.5'])]
# from sklearn.model_selection import train_test_split

# X  = dataTrain.drop(["PM2.5","device_id","Date","Time","Year","Month","Day"],axis=1)

# y = dataTrain['PM2.5']

# dataTest  = dataTest.drop(["device_id","Date","Time","Year","Month","Day"],axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)



from sklearn.model_selection import train_test_split

X  = dataTrain.drop(["PM2.5","Date","Time","Year","Month","Day"],axis=1)

y = dataTrain['PM2.5']

dataTest  = dataTest.drop(["device_id","Date","Time","Year","Month","Day",'PM2.5'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)

myx_test=X_test

X_train=X_train.drop(["device_id"],axis=1)

X_test=X_test.drop(["device_id"],axis=1)
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

xgbrModel=xgb.XGBRegressor(n_estimators=100,n_jobs=-1,max_depth=9,learning_rate=0.1)

xgbrModel.fit(X_train,y_train)

xbPreds = xgbrModel.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt

print("Score: ",xgbrModel.score(X_test, y_test))

print("MAE: ",mean_absolute_error(y_test,xbPreds))

print("RMSE: ",sqrt(mean_squared_error(y_test,xbPreds)))
#取每個觀測站平均

yPreds=np.zeros(252)

for i in range(0,252):

    newOutput=y_test[myx_test['device_id']==submission.iloc[i]['device_id']]

    yPreds[i]=newOutput.mean()



myPreds=np.zeros(252)

for i in range(0,252):

    newOutput=xbPreds[myx_test['device_id']==submission.iloc[i]['device_id']]

    myPreds[i]=newOutput.mean()

    

# print("yPred",yPreds)

# print("myPred",myPreds)

#計算平均誤差

print("rmse",sqrt(mean_squared_error(yPreds,myPreds)))

print("mbe",mean_absolute_error(yPreds,myPreds))
xbPreds = xgbrModel.predict(dataTest)

# Generate Submission File 

arr=np.zeros(252)

for i in range(0,252):

    newOutput=xbPreds[test['device_id']==submission.iloc[i]['device_id']]

    arr[i]=newOutput.mean()

newsSbmission=submission

newsSbmission["pred_pm25"]=arr

newsSbmission.to_csv("submission.csv", index=False)