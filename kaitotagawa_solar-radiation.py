# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import and set basic parameters / variables 

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

data = "/kaggle/input/SolarEnergy/SolarPrediction.csv"

csvData = pd.read_csv(data)

#どのようなパラメーターがあるかまず見る。

csvData.head()
#放射量と温度の関係性を見る。

jointTempRad = sns.jointplot(x="Radiation", y="Temperature", data=csvData)

csvData['timeConv'] = pd.to_datetime(csvData['Time'], format = '%H:%M:%S')



#新しい col month

csvData['month'] = pd.to_datetime(csvData['UNIXTime'].astype(int), unit='s').dt.month

#print(csvData['month'])



#新しい col day

csvData['day'] = pd.to_datetime(csvData['UNIXTime'].astype(int), unit = 's').dt.day



#新しい　col hour　　

csvData['hour'] = pd.to_datetime(csvData['timeConv'], format = '%H:%M:%S').dt.hour

#print(csvData['hour'])



#新しい col totalSunTime　一日、日が出ていた時間のトータル。

csvData['totalSunTime'] = pd.to_datetime(csvData['TimeSunSet'], format = '%H:%M:%S').dt.hour - pd.to_datetime(csvData['TimeSunRise'], format = '%H:%M:%S').dt.hour 



#Radiationが10以下のデータは日が出てない時のデータだと思われるので消す。

csvData = csvData[csvData['Radiation'] > 10]

#放射量と時間の関係性を見る。

ax = plt.axes()

sns.barplot(x="hour", y='Radiation', data=csvData, ax = ax)

ax.set_title('Mean Radiation by Hour')

plt.show()
#新しく作られたパラメーターtotalSunTimeと放射量の関係性を見る。

ax2 = plt.axes()

sns.barplot(x = "totalSunTime", y = 'Radiation', data = csvData, ax = ax2)

#increase in totalSunTime indicates more radiation 
#放射量と月の関係性を見る。

ax3 = plt.axes()

sns.barplot(x="month", y='Radiation', data=csvData, ax = ax3)
#モデル構築を始める。

from sklearn.metrics import r2_score

#yを放射量　Xは放射量と関係性がありそうなパラメーターを使う。

y = csvData['Radiation']

X = csvData[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'totalSunTime', 'hour', 'month', 'day']]



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor

from sklearn.metrics import r2_score

import xgboost as xgb



# 検証用のデータを20％とっておき、あとは学習用のデータとして使う。

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=1)



#回帰モデルが入ったリストを　initialise　する。

regression = [LinearRegression(), Ridge(), DecisionTreeRegressor(), RandomForestRegressor(), 

              GradientBoostingRegressor(), AdaBoostRegressor(), xgb.XGBRegressor(objective="reg:linear", random_state=42)]



#scoresというリストをdefineしておく。

scores = []

num = 0

#ループで、regression に入っているすべての回帰モデルを試して、そのスコアを　scoresリスト　に入れていく。

#scores に入る tuple　は　（モデルの頭文字3文字、trainのr2スコア、testのr2スコア）で構成される。

for i in regression:

    i.fit(X_train, y_train)

    scores.append((str(i)[ :3], r2_score(y_train, i.predict(X_train)), r2_score(y_test, i.predict(X_test))))

    print(i.score(X_test, y_test))



#スコアをprintする。

print(scores)

pd.Series(scores)

pd.DataFrame.from_records(scores, columns = ['Method', 'Test', 'Train']) 