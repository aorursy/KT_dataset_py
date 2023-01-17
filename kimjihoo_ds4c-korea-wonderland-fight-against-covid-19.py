from IPython.display import YouTubeVideo

YouTubeVideo('xbbU1PBemC4', width="640", height="360")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/coronavirusdataset'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/coronavirusdataset/'



case = p_info = pd.read_csv(path+'Case.csv')

p_info = pd.read_csv(path+'PatientInfo.csv')

p_route = pd.read_csv(path+'PatientRoute.csv')

time = pd.read_csv(path+'Time.csv')

t_age = pd.read_csv(path+'TimeAge.csv')

t_gender = pd.read_csv(path+'TimeGender.csv')

t_provin = pd.read_csv(path+'TimeProvince.csv')

region = pd.read_csv(path+'Region.csv')

weather = pd.read_csv(path+'Weather.csv')

search = pd.read_csv(path+'SearchTrend.csv')

floating = pd.read_csv(path+'SeoulFloating.csv')



dt = pd.read_csv('/kaggle/input/covid19-screening-center/dt.csv')

kor_province = '/kaggle/input/covid19-screening-center/TL_SCCO_SIG_WGS84.json'
# Make new test/confirmed/released/deceased from accumulated counts

new_test = [1]

new_negative = [0]

new_confirmed = [1]

new_released = [0]

new_deceased = [0]



for i in range(len(time['date'])-1):

    new_test.append(time['test'][i+1] - time['test'][i])

    new_negative.append(time['negative'][i+1] - time['negative'][i])

    new_confirmed.append(time['confirmed'][i+1] - time['confirmed'][i])

    new_released.append(time['released'][i+1] - time['released'][i])

    new_deceased.append(time['deceased'][i+1] - time['deceased'][i])

    

time['new_test'] = new_test

time['new_negative'] = new_negative

time['new_confirmed'] = new_confirmed

time['new_released'] = new_released

time['new_deceased'] = new_deceased
plt.figure(figsize=(12, 8))

plt.title('Korea\'s Testing and Diagnostic Capabilities', fontsize=24)



plt.plot(time.date, time.test, label='acc_test')

plt.plot(time.date, time.negative, label='acc_negative')

plt.plot(time.date, time.confirmed, label='acc_confirmed')



plt.scatter(list(time['date'])[-1], list(time['test'])[-1])

plt.text(list(time['date'])[-3], list(time['test'])[-1]+7000, list(time['test'])[-1])

plt.scatter(list(time['date'])[-1], list(time['negative'])[-1])

plt.text(list(time['date'])[-3], list(time['negative'])[-1]+7000, list(time['negative'])[-1])

plt.scatter(list(time['date'])[-1], list(time['confirmed'])[-1])

plt.text(list(time['date'])[-2], list(time['confirmed'])[-1]+7000, list(time['confirmed'])[-1])



plt.axvline('2020-02-04', color='r', alpha = 1)

plt.text('2020-02-04', 350000, '1st EUA', fontsize=12)

plt.axvline('2020-02-12', color='r', alpha = 0.8)

plt.text('2020-02-12', 340000, '2nd EUA', fontsize=12)

plt.axvline('2020-02-27', color='r', alpha = 0.6)

plt.text('2020-02-27', 330000, '3rd & 4th EUA', fontsize=12)

plt.axvline('2020-03-13', color='r', alpha = 0.4)

plt.text('2020-03-13', 320000, '5th EUA', fontsize=12)



plt.xticks(['2020-01-20','2020-01-27',

            '2020-02-03','2020-02-10','2020-02-17','2020-02-24',

            '2020-03-02','2020-03-09','2020-03-16','2020-03-23','2020-03-30'])

plt.xticks(rotation=30)



plt.legend(fontsize=12)



plt.show()
plt.figure(figsize=(12, 8))

plt.title('Korea\'s Testing and Diagnostic Capabilities', fontsize=24)



plt.plot(time.date, time.new_test, label='new_test')

plt.plot(time.date, time.new_negative, label='new_negative')

plt.plot(time.date, time.new_confirmed, label='new_confirmed')



plt.axvline('2020-02-04', color='r', alpha = 1)

plt.text('2020-02-04', 3500, '1st EUA', fontsize=12)

plt.axvline('2020-02-12', color='r', alpha = 0.8)

plt.text('2020-02-12', 3000, '2nd EUA', fontsize=12)

plt.axvline('2020-02-27', color='r', alpha = 0.6)

plt.text('2020-02-27', 2500, '3rd & 4th EUA', fontsize=12)

plt.axvline('2020-03-13', color='r', alpha = 0.4)

plt.text('2020-03-13', 2000, '5th EUA', fontsize=12)



plt.xticks(['2020-01-20','2020-01-27',

            '2020-02-03','2020-02-10','2020-02-17','2020-02-24',

            '2020-03-02','2020-03-09','2020-03-16','2020-03-23','2020-03-30'])

plt.xticks(rotation=30)



plt.legend(fontsize=12)



plt.show()
import folium 

import json

from folium import plugins



m = folium.Map([36, 128], zoom_start=7) 



plugins.Fullscreen(position='topright',  # Full screen

                   title='Click to Expand', 

                   title_cancel='Click to Exit', 

                   force_separate_button=True).add_to(m)



plugins.MousePosition().add_to(m) ## you can easily get coordinates.



with open('/kaggle/input/covid19-screening-center/TL_SCCO_SIG_WGS84.json',mode='rt',encoding='utf-8') as f:

    geo = json.loads(f.read())

    f.close()

    

folium.GeoJson(

    geo,

    name='seoul_municipalities'

).add_to(m)



for i in range(len(dt)):



    location = (dt['latitude'][i],dt['longitude'][i])

    folium.Marker(location, popup = dt['province'][i], icon = folium.Icon(color = 'gray')).add_to(m)



m