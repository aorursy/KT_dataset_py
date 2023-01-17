# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.



import datetime, time
# 犯罪データ取り込み

df_crime = pd.read_csv('../input/crime-dataset/Chicago_Crimes_2012_to_2017.csv')

# 不要な列を削除

#df_crime = df_crime.drop(['Block', 'IUCR', 'Description', 'Location Description', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area','FBI Code', 'Updated On', 'Latitude', 'Longitude', 'Location'], axis=1)

df_crime = df_crime[['ID', 'Date', 'Primary Type', 'Arrest', 'X Coordinate', 'Y Coordinate', 'Year']]

# 空白を含まない列名に変更

df_crime = df_crime.rename(columns={'Primary Type': 'Type','X Coordinate': 'Coord_X', 'Y Coordinate': 'Coord_Y'})

# NaNデータ削除

df_crime = df_crime.dropna(axis = 0)

# 2013～2016年のデータのみ抽出（天候データは2012/10月から／最後に2017/1月のデータが少しあり）

df_crime = df_crime[(df_crime['Year'] >= 2013) & (df_crime['Year'] < 2017)]

# 座標の無いデータを削除

df_crime = df_crime[df_crime['Coord_X'] > 0]



df_crime.head(3)
# UNIX TIMEを計算

df_crime['UNIX_TIME'] = [int(time.mktime(time.strptime(d, '%m/%d/%Y %I:%M:%S %p'))) for d in df_crime['Date']]

# UNIX TIMEでソート

df_crime = df_crime.sort_values('UNIX_TIME')

df_crime.head()
# 気温データ取り込み

df_temp = pd.read_csv('../input/historical-hourly-weather-data/temperature.csv')

df_temp = df_temp[['datetime', 'Chicago']]

df_temp = df_temp.dropna()

# 24時間列

df_temp['Hour'] =  [x[11:13] for x in df_temp['datetime']]

# UNIX TIMEを計算

df_temp['UNIX_TIME'] = [int(time.mktime(time.strptime(d, '%Y-%m-%d %H:%M:%S'))) for d in df_temp['datetime']]

# UNIX TIMEでソート

df_temp = df_temp.sort_values('UNIX_TIME')

# 2013～2016年のデータのみ抽出

lower = int(time.mktime(time.strptime('2013-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))

upper = int(time.mktime(time.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))

df_temp = df_temp[(df_temp['UNIX_TIME'] >= lower) & (df_temp['UNIX_TIME'] < upper) ]

# ケルビンを摂氏に変換

df_temp['Celsius'] = [f - 273.15 for f in df_temp['Chicago']]



df_temp.head(3)

# 温度データの分布

plt.hist(df_temp['Celsius'], bins=100)
# 湿度データ取り込み

df_humi = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv')

df_humi = df_humi[['datetime', 'Chicago']]

df_humi = df_humi.dropna()

# 24時間列

df_humi['Hour'] =  [x[11:13] for x in df_humi['datetime']]

# UNIX TIMEを計算

df_humi['UNIX_TIME'] = [int(time.mktime(time.strptime(d, '%Y-%m-%d %H:%M:%S'))) for d in df_humi['datetime']]

# UNIX TIMEでソート

df_humi = df_humi.sort_values('UNIX_TIME')

# 2013～2016年のデータのみ抽出

lower = int(time.mktime(time.strptime('2013-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))

upper = int(time.mktime(time.strptime('2017-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')))

df_humi = df_humi[(df_humi['UNIX_TIME'] >= lower) & (df_humi['UNIX_TIME'] < upper) ]



df_humi.head(3)
#湿度データの分布 100%は除外しよう

plt.hist(df_humi['Chicago'], bins=100)
# 時刻と気温の関係

def time_temp():

    cold = int(np.min(df_temp['Celsius']))

    hot = int(np.max(df_temp['Celsius']))

    print('{:}℃ ～ {:}℃'.format(cold, hot))

    temp_range = int(abs(cold)+hot+1)

    arr = np.zeros((temp_range, 24+1)) # hour, temp

    for idx, row in df_temp.iterrows():

        hour = int(row['Hour'])

        # マイナスを考慮して底上げ

        cel = int(row['Celsius'] + abs(cold))

        arr[cel][0] = cel - abs(cold)

        arr[cel][hour+1] += 1

    # ヒートマップ

    sns.heatmap(arr[:,1:], xticklabels=True, yticklabels=arr[:,0])

    return pd.DataFrame(arr)
# 時刻と湿度の関係

def time_humi():

    cold = int(np.min(df_humi['Chicago']))

    hot = int(np.max(df_humi['Chicago']))

    print('{:}% ～ {:}%'.format(cold, hot))

    arr = np.zeros((100, 24+1)) #100%は除外なので

    for idx, row in df_humi.iterrows():

        hour = int(row['Hour'])

        hum = int(row['Chicago'])

        if hum < 100:

            arr[hum][0] = hum

            arr[hum][hour+1] += 1

    # ヒートマップ

    sns.heatmap(arr[:,1:], xticklabels=True, yticklabels=arr[:,0])

    return pd.DataFrame(arr)
# 気温と発生件数

# 先にdf_time_temp = time_temp()しておくこと！

# 引数1: 犯罪タイプ，''で全て

# 引数2: 気温の出現頻度による補正有無

def heatmap_temp(crime_type, corr):

    cold = int(np.min(df_temp['Celsius']))

    hot = int(np.max(df_temp['Celsius']))

    print('{:}℃ ～ {:}℃'.format(cold, hot))

    temp_range = int(abs(cold)+hot+1)

    if len(crime_type) > 0:

        df_crime_type = df_crime[df_crime['Type'] == crime_type]

    else:

        df_crime_type = df_crime

    arr = np.zeros((temp_range, 24+1)) # hour, temp

    for idx, row in df_temp.iterrows():

        hour = int(row['Hour'])

        ut = row['UNIX_TIME']

        crime = df_crime_type[(df_crime_type['UNIX_TIME'] >= ut) & (df_crime_type['UNIX_TIME'] < ut + 3600)]

        # マイナスを考慮して底上げ

        cel = int(row['Celsius'] + abs(cold))

        arr[cel][0] = cel - abs(cold)

        if corr:

            arr[cel][hour+1] += len(crime) / df_time_temp.iloc[cel, hour+1]

        else:

            arr[cel][hour+1] += len(crime)

    # check

    if not corr:

        print('crime={:}, calculated={:}'.format(len(df_crime_type),int(sum(sum(arr[:,1:])))))

        if len(df_crime_type) == int(sum(sum(arr[:,1:]))):

            print('check OK!')

        else:

            print('something wrong...')

    # ヒートマップ

    sns.heatmap(arr[:,1:], xticklabels=True, yticklabels=arr[:,0])
# 湿度と発生件数

# 先にdf_time_humi = time_humi()しておくこと！

# 引数1: 犯罪タイプ，''で全て

# 引数2: 気温の出現頻度による補正有無

def heatmap_humi(crime_type, corr):

    cold = int(np.min(df_humi['Chicago']))

    hot = int(np.max(df_humi['Chicago']))

    print('{:}% ～ {:}%'.format(cold, hot))

    if len(crime_type) > 0:

        df_crime_type = df_crime[df_crime['Type'] == crime_type]

    else:

        df_crime_type = df_crime

    arr = np.zeros((101, 24+1)) # hour, temp

    for idx, row in df_humi.iterrows():

        hour = int(row['Hour'])

        ut = row['UNIX_TIME']

        crime = df_crime_type[(df_crime_type['UNIX_TIME'] >= ut) & (df_crime_type['UNIX_TIME'] < ut + 3600)]

        hum = int(row['Chicago'])

        if hum < 100: #100%は除外なので

            arr[hum][0] = hum

            if corr:

                arr[hum][hour+1] += len(crime) / df_time_humi.iloc[hum, hour+1]

            else:

                arr[hum][hour+1] += len(crime)

    # check

    if not corr:

        print('crime={:}, calculated={:}'.format(len(df_crime_type),int(sum(sum(arr[:,1:])))))

        if len(df_crime_type) == int(sum(sum(arr[:,1:]))):

            print('check OK!')

        else:

            print('something wrong...')

    # ヒートマップ

    sns.heatmap(arr[:,1:], xticklabels=True, yticklabels=arr[:,0])
# 犯罪タイプ

types = df_crime['Type'].unique()

types
# 時刻と温度の関係

df_time_temp = time_temp()

df_time_temp.head()
# ヒートマップ 温度出現頻度の補正無し

heatmap_temp('', False)

# ヒートマップ 温度出現頻度の補正有り

heatmap_temp('BATTERY', True)
# ヒートマップ 温度出現頻度の補正有り

heatmap_temp('ASSAULT', True)
# 時刻と湿度の関係

df_time_humi = time_humi()

#df_time_humi.head()
heatmap_humi('', False)
heatmap_humi('', True)
heatmap_humi('BATTERY', True)
heatmap_humi('ASSAULT', True)