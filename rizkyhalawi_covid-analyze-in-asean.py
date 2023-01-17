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

record_date = '2020-08-17'

covid_url = 'https://covid19-api.org/api/status?date='+ record_date

df_covid_worldwide = pd.io.json.json_normalize(get_json(covid_url))



print(df_covid_worldwide.head())



#Merubah Format date

df_covid_worldwide['last_update'] = pd.to_datetime(df_covid_worldwide['last_update'], format='%Y-%m-%d %H:%M:%S')

df_covid_worldwide['last_update'] = df_covid_worldwide['last_update'].apply(lambda x: x.date())



#Mengambil data Countries

countries_url = 'https://covid19-api.org/api/countries'

df_countries = pd.io.json.json_normalize(get_json(countries_url))

df_countries = df_countries.rename(columns={'alpha2': 'country'})[['name','country']]



print(df_countries.head())



#Merge Covid19 Data dan Countries

df_covid_denormalized = pd.merge(df_covid_worldwide, df_countries, on='country')



print(df_covid_denormalized.head())



#MENGHITUNG FATALITY RATIO

df_covid_denormalized['fatality_ratio'] = df_covid_denormalized['deaths']/df_covid_denormalized['cases']

#Negara-negara dengan Fatality Ratio Tertinggi

df_top_20_fatality_rate = df_covid_denormalized.sort_values(by='fatality_ratio', ascending=False).head(20)

print(df_top_20_fatality_rate)

#Import Library Visualisasi dan Visualisasi Negara dengan Fatality Ratio Tertinggi

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))

x = df_top_20_fatality_rate['name']

y = df_top_20_fatality_rate['fatality_ratio']

plt.bar(x,y)

plt.xlabel('Country Name')

plt.ylabel('Fatality Rate')

plt.title('Top 20 Highest Fatality Rate Countries')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()



#Menggabungkan DataFrame

countries = ['ID','MY','SG','TH','VN']

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



#Merge Data Covid19 Dengan Data Country

df_covid_timeline_denormalized = pd.merge( df_covid_timeline_merged, df_countries, on='country')



#kasus COVID di ASEAN 2019

import datetime

df_covid_timeline_denormalized = df_covid_timeline_denormalized[(df_covid_timeline_denormalized['last_update'] >= datetime.date(2020, 3, 1))]



import matplotlib.pyplot as plt

plt.clf()

countries = ['ID','MY','SG','TH','VN']

for country in countries:

	country_data = df_covid_timeline_denormalized['country']== country

	x = df_covid_timeline_denormalized[country_data]['last_update']

	y = df_covid_timeline_denormalized[country_data]['cases']

	plt.plot(x, y, label = country)



plt.legend()

plt.xlabel('Record Date')

plt.ylabel('Total Cases')

plt.title('Asean Covid19 Cases Comparison')

plt.show()


