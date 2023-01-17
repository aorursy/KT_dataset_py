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
#Library yg di butuhkan

import json

import numpy as np

import pandas as pd

import requests



#Membuat Fungsi get API

def get_json(api_url):

	response = requests.get(api_url)

	if response.status_code == 200:

		return json.loads(response.content.decode('utf-8'))

	else:

		return None



#Memanggil API Covid19

record_date = '2020-09-20'

covid_url = 'https://covid19-api.org/api/status?date='+ record_date

df_covid_worldwide = pd.io.json.json_normalize(get_json(covid_url))

import datetime

df_covid_worldwide['last_update_month'] = df_covid_worldwide['last_update'].apply

(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S").strftime('%Y-%m'))



print(df_covid_worldwide.head())



countries = ['JP']

i = 0

for country in countries:

	covid_timeline_url = 'https://covid19-api.org/api/timeline/'+country

	df_covid_timeline = pd.io.json.json_normalize(get_json(covid_timeline_url))

	df_covid_timeline['last_update'] = pd.to_datetime(df_covid_timeline['last_update'], format='%Y-%m-%dT%H:%M:%S')

	df_covid_timeline['last_update'] = df_covid_timeline['last_update'].apply(lambda x: x.date())

	if i==0:

		df_covid_timeline_merged = df_covid_timeline

	else:

		df_covid_timeline_merged = df_covid_timeline.append(df_covid_timeline_merged, ignore_index=True)

	i=i+1



print(df_covid_timeline_merged.head())



#MENGHITUNG FATALITY RATIO

df_covid_timeline_merged['fatality_ratio'] = df_covid_timeline_merged['deaths']/df_covid_timeline_merged['cases']

#RATA_RATA Fatality Ratio DI JEPANG

df_japan_fatality_rate = df_covid_timeline_merged.sort_values(by='last_update', ascending=False).head(20)

df_Japan_fatality_rate_mean = df_covid_timeline_merged['fatality_ratio'].mean()

#Menghitung Cure RAtio

df_covid_timeline_merged['cure_ratio'] = df_covid_timeline_merged['recovered']/df_covid_timeline_merged['cases']

df_japan_cure_rate = df_covid_timeline_merged.sort_values(by='last_update', ascending=False).head(20)

df_Japan_cure_rate_mean = df_covid_timeline_merged['cure_ratio'].mean()





print(df_japan_fatality_rate.head())

print(df_japan_cure_rate.head())

print("FATALITY RATIO RATA DI JEPANG SAMPAI BULAN SEPTEMBER ADALAH =", df_Japan_fatality_rate_mean)

print("CURE RATIO RATA DI JEPANG SAMPAI BULAN SEPTEMBER ADALAH =", df_Japan_cure_rate_mean)



#kasus COVID di ASEAN 2019

import datetime

df_covid_timeline_merged = df_covid_timeline_merged[(df_covid_timeline_merged['last_update'] >= datetime.date(2020, 3, 1))]

df_covid_timeline_merged['fatality_ratio%']= df_covid_timeline_merged['fatality_ratio']*100

df_covid_timeline_merged['cure_ratio%']=df_covid_timeline_merged['cure_ratio']*100

import matplotlib.pyplot as plt

#FATALITY RATIO

fig = plt.figure(figsize=(15, 5))

df_covid_timeline_merged.groupby(['last_update'])['fatality_ratio%'].sum().plot(color='red', linestyle='-.', linewidth=2)

plt.title('Fatality Ratio COVID 19 in Japan From March to September', loc='center', pad=40, fontsize=20, color='#fcc203')

plt.xlabel('Month', fontsize=15)

plt.ylabel('Fatality Ratio (in %)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1).astype(int))

plt.show()



#CURE RATIO

fig = plt.figure(figsize=(15, 5))

df_covid_timeline_merged.groupby(['last_update'])['cure_ratio%'].sum().plot(color='#00a5f7', linestyle='-.', linewidth=2)

plt.title('Recovery rate COVID 19 in Japan From March to September', loc='center', pad=40, fontsize=25, color='Green')

plt.xlabel('Month', fontsize=15)

plt.ylabel('Recovery rate (in %)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1).astype(int))

plt.show()



#PENAMBAHAN KASUS PER_HARI

df_covid_timeline_merged['cases_added'] = (df_covid_timeline_merged['cases'].diff(-1).fillna(0))

df_japan_cases_added = df_covid_timeline_merged.sort_values(by='last_update', ascending=False).head(20)

print(df_japan_cases_added.head())



print('RATA-RATA PENAMBAHAN KASUS\nCOVID DI JEPANG =', df_japan_cases_added['cases_added'].max())



#grafik Penambahan Kasus di Japan

plt.clf()

plt.figure(figsize=(15, 5))

df_covid_timeline_merged.groupby(['last_update'])['cases_added'].sum().plot(color='#fcba03', linestyle='-', linewidth=2)

plt.title('Additional Cases COVID 19 in Japan From March to September', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Month', fontsize=15)

plt.ylabel('Additional cases', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1).astype(int))

plt.show()



#kasus COVID di JEPANG dari maret hingga  september

import matplotlib.pyplot as plt

plt.clf()

plt.figure(figsize=(15, 5))

df_covid_timeline_merged.groupby(['last_update'])['cases'].sum().plot(color='green', linestyle='-.', linewidth=2)

df_covid_timeline_merged.groupby(['last_update'])['recovered'].sum().plot(color='red', linestyle='-.', linewidth=2)

plt.title('Cases of COVID 19 in Japan until September 2020', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Month', fontsize=15)

plt.ylabel('Total Cases (in thousand)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.legend(shadow=True, ncol=1, title='Data')

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000).astype(int))

plt.show()





#KASUS COVID PER PREFEKTUR

covid_url = 'https://covid19-japan-web-api.now.sh/api//v1/prefectures'

df_covid_japan = pd.io.json.json_normalize(get_json(covid_url))

#Merubah Format date

df_covid_japan['last_updated.cases_date'] = pd.to_datetime(df_covid_japan['last_updated.cases_date'], format='%Y-%m-%d %H:%M:%S')

df_covid_japan['last_updated.cases_date'] = df_covid_japan['last_updated.cases_date'].apply(lambda x: x.date())





#Mengambil data top 10 prefecture

df_top_10_prefecture = df_covid_japan[["name_ja","name_en", "cases","deaths"]].sort_values(by='cases', ascending=False).head(10)

print(df_top_10_prefecture)



#Grafik KASUS COVID PER PREFECTURE

plt.clf()

plt.figure(figsize=(10, 5))

df_top_10_prefecture.groupby(['name_en'])['cases'].sum().sort_values(ascending=False).plot(kind='bar',color='green')

df_top_10_prefecture.groupby(['name_en'])['deaths'].sum().sort_values(ascending=False).plot(kind='bar',color='blue')

plt.title('Top 10 Prefecture COVID 19 Cases in Japan',loc='center',pad=30, fontsize=15, color='blue')

plt.xlabel('Prefecture', fontsize = 15)

plt.ylabel('Total Cases (in Thousand)', fontsize = 15)

plt.ylim(ymin=0)

plt.legend(bbox_to_anchor=(1, 1), shadow=True, ncol=1, title='Cases and Deaths')

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000).astype(int))

plt.xticks(rotation=45)

plt.show()


