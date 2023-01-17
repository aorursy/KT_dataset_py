# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import requests
from bs4 import BeautifulSoup as bs

def month_to_int(month_tgt):
    if month_tgt == 'January': return 1
    if month_tgt == 'February': return 2
    if month_tgt == 'March': return 3
    if month_tgt == 'April': return 4
    if month_tgt == 'May': return 5
    if month_tgt == 'June': return 6
    if month_tgt == 'July': return 7
    if month_tgt == 'August': return 8
    if month_tgt == 'September': return 9
    if month_tgt == 'October': return 10
    if month_tgt == 'November': return 11
    if month_tgt == 'December': return 12

original = pd.DataFrame()
for target in ['2011', '2012']:
    req = requests.get('https://www.calendar-365.com/holidays/%s.html' % target)
    bs1 = bs(req.text)
    bs2 = bs1.find('div', class_ = 'table-scroll-view')
    head = bs2.find('thead').find('th')
    body_list = bs2.find('tbody').find_all('tr')
    year_list, day_list, month_list, holiday_list = [], [], [], []
    for line in body_list:
        for n, item in enumerate(line.find_all('td')):
            if n == 0: 
                month, day, year = item.text.split(' ')
                year_list.append(year)
                day_list.append(day)
                month_list.append(month_to_int(month))
            elif n == 1: holiday_list.append(item.text)
            else: break
    df = pd.DataFrame()
    df['year'] = year_list
    df['day'] = day_list
    df['month'] = month_list
    df['holiday_name'] = holiday_list
    original = pd.concat([original, df])
original['holiday'] = 1
original['ymd'] = original['year'].astype(str) +"-"+ original['month'].astype(str) +"-"+ original['day'].astype(str)
original.to_csv('real_holiday.csv', index=False)
original.head()
from datetime import datetime
import time
import json

start_date = datetime.strptime('2011-01-01', '%Y-%m-%d')
str_list = []
for i in range(370):
    end_date = start_date + pd.Timedelta(days=1)
    base = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=&key=3e325be0b3154bc19ee95130201502&q=20001&format=json&extra=localObsTime'
    start = '&date=%s-%s-%s' % (start_date.year, start_date.month, start_date.day)
    end = '&enddate=%s-%s-%s' % (end_date.year, end_date.month, end_date.day)
    suffix = '&show_comments=no&tp=1'
    url = base + start + end + suffix
    str_list.append(requests.get(url).text)
    start_date = start_date + pd.Timedelta(days=2)
    time.sleep(0.1)

all_req = pd.DataFrame()
for one_req in str_list:
    double_date = json.loads(one_req)
    single_req_df = pd.DataFrame()
    for single_day in double_date['data']['weather']:
        date = single_day['date']
        by_hour = single_day['hourly']
        day_df = pd.DataFrame()
        for h_iter in by_hour:
            single_hour = {'date' : single_day['date']}
            weather_interest = ['time', 'tempC', 'windspeedKmph', 'weatherCode', 'weatherDesc', 
                                'precipMM', 'humidity', 'cloudcover', 'WindGustKmph']
            for wi in weather_interest:
                single_hour[wi] = h_iter[wi]
            day_df = pd.concat([day_df, pd.DataFrame(single_hour, index=[0])], sort=False)
        single_req_df = pd.concat([single_req_df, day_df], sort=False)
    all_req = pd.concat([all_req, single_req_df], sort=False)
all_req.to_csv('real_weather.csv', index=False)
all_req.head()