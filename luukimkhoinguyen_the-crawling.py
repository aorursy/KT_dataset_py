# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import requests

from bs4 import BeautifulSoup

import itertools



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
page1 = requests.get('https://www.timeanddate.com/weather/australia/melbourne/historic?hd=20200822')

page2 = requests.get('https://www.timeanddate.com/weather/australia/melbourne/historic?hd=20200821')

page3 = requests.get('https://www.timeanddate.com/weather/australia/melbourne/historic?hd=20200820')

page4 = requests.get('https://www.timeanddate.com/weather/australia/melbourne/historic?hd=20200819')

page5 = requests.get('https://www.timeanddate.com/weather/australia/melbourne/historic?hd=20200818')
list_temp = []

list_time = []

list_wind = []

list_baro = []

list_humi = []

date = []
def git_gud(url, num):

    soup = BeautifulSoup(url.content, 'html.parser')

    table = soup.find(id='wt-his')

    trs = table.find_all('tr')

    temp = []

    time = []

    wind = []

    baro = []

    humi = []

    

    for i in trs:

        ths = i.find_all('th')

        tds = i.find_all('td')

        if len(tds)>1:

            time.append(ths[0].text.strip())

            temp.append(tds[1].text.replace('\xa0°F', ''))

            wind.append(tds[3].text.strip().replace(' mph', ''))

            baro.append(tds[6].text.strip().replace(' "Hg', ''))

            humi.append(tds[5].text.replace('%', ''))

            date.append(num)

            

    time[0] = time[0][:-11]

    list_temp.append(temp)

    list_time.append(time)

    list_wind.append(wind)

    list_baro.append(baro)

    list_humi.append(humi)
git_gud(page1, 22)

git_gud(page2, 21)

git_gud(page3, 20)

git_gud(page4, 19)

git_gud(page5, 18)
final_temp = list(itertools.chain.from_iterable(list_temp))

final_time = list(itertools.chain.from_iterable(list_time))

final_wind = list(itertools.chain.from_iterable(list_wind))

final_baro = list(itertools.chain.from_iterable(list_baro))

final_humi = list(itertools.chain.from_iterable(list_humi))
d = {'date': date, 'time': final_time, 'temp': final_temp, 'wind': final_wind, 'humidity': final_humi, 'barometer': final_baro}

df = pd.DataFrame(data=d)

df
df['temp'] = df['temp'].astype(int)

df['date'] = df['date'].astype(int)

df['wind'] = df['wind'].astype(int)

df['humidity'] = df['humidity'].astype(int)

df['barometer'] = df['barometer'].astype(float)



df['temp'] = (df['temp'] -32)*5/9

df['wind'] = df['wind'] * 1.609344

df['barometer'] = df['barometer'] * 33.86388666666671
df['temp'] = df['temp'].astype(int)

df['barometer'] = df['barometer'].astype(int)

df['wind'] = df['wind'].astype(int)

df.dtypes
df.to_csv('melbourne.18_08_2020.22_08_2020.csv', index=False)
def compute_max_temp(df):

    return df.groupby('date').temp.max().tolist()



compute_max_temp(df)