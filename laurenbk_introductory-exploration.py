# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns



import mpl_toolkits.basemap

from mpl_toolkits.basemap import Basemap



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
air_reserve = pd.read_csv('../input/air_reserve.csv')

air_store_info = pd.read_csv('../input/air_store_info.csv')

air_visit_data = pd.read_csv('../input/air_visit_data.csv')

date_info = pd.read_csv('../input/date_info.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')

store_id_relation = pd.read_csv('../input/store_id_relation.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
air_reserve.head()
hpg_reserve.head()
air_store_info.head()
hpg_store_info.head()
date_info.head()
air_visit_data.head()
sample_submission.head()
store_id_relation.head()
air_reserve.visit_datetime = pd.to_datetime(air_reserve.visit_datetime)

air_reserve.reserve_datetime = pd.to_datetime(air_reserve.reserve_datetime)

hpg_reserve.visit_datetime = pd.to_datetime(hpg_reserve.visit_datetime)

hpg_reserve.reserve_datetime = pd.to_datetime(hpg_reserve.reserve_datetime)

air_visit_data.visit_date = pd.to_datetime(air_visit_data.visit_date)

date_info.calendar_datetime = pd.to_datetime(date_info.calendar_date)
plt.figure(figsize= (10,7))

goldenweek1 = pd.date_range('2016-04-29','2016-5-5')

goldenweek2 = pd.date_range('2017-04-29','2017-5-7')

plt.subplot(2,1,1)

plt.axvspan(goldenweek1[0],goldenweek1[-1], color = 'y', 

            alpha=0.5, label = 'Golden week')

plt.axvspan(goldenweek2[0],goldenweek2[-1], color = 'y', 

            alpha=0.5)

plt.scatter(list(date_info.calendar_datetime.loc[date_info.holiday_flg == 1]),

            date_info.holiday_flg.loc[date_info.holiday_flg == 1], 

            color = 'r', label = 'Holidays', marker = '^')

air_daily_reservations = air_reserve[['visit_datetime','reserve_visitors']].set_index('visit_datetime').resample('d').sum()

plt.plot(air_daily_reservations.index, air_daily_reservations.reserve_visitors, alpha = 0.5, label = 'AIR')

plt.xlabel('Reservation Date')

plt.ylabel('Number of visitors')

plt.legend()

plt.subplot(2,1,2)

plt.axvspan(goldenweek1[0],goldenweek1[-1], color = 'y', 

            alpha=0.5, label = 'Golden week')

plt.axvspan(goldenweek2[0],goldenweek2[-1], color = 'y', 

            alpha=0.5)

plt.scatter(list(date_info.calendar_datetime.loc[date_info.holiday_flg == 1]),

            date_info.holiday_flg.loc[date_info.holiday_flg == 1], 

            color = 'r', label = 'Holidays', marker = '^')

hpg_daily_reservations = hpg_reserve[['visit_datetime','reserve_visitors']].set_index('visit_datetime').resample('d').sum()

plt.plot(hpg_daily_reservations.index, hpg_daily_reservations.reserve_visitors, alpha = 0.5, label = 'HPG')

plt.xlabel('Reservation Date')

plt.ylabel('Number of visitors')

plt.legend()
air_reserve['dateDiff'] = air_reserve.visit_datetime - air_reserve.reserve_datetime

hpg_reserve['dateDiff'] = hpg_reserve.visit_datetime - hpg_reserve.reserve_datetime



plt.figure(figsize= (10,7))

plt.subplot(2,2,1)

sample = (air_reserve.dateDiff.dt.total_seconds()/3600).value_counts()

sample.plot(kind = 'bar', label = 'AIR bookings')

plt.legend()

plt.xlabel('Hours between booking and reservation')

plt.subplot(2,2,2)

sample = (hpg_reserve.dateDiff.dt.total_seconds()/3600).value_counts()

sample.plot(kind = 'bar', label = 'HPG bookings')

plt.xlabel('Hours between booking and reservation')

plt.legend()

plt.subplot(2,2,3)

sample = (air_reserve.dateDiff.dt.total_seconds()/3600).value_counts()

sample = sample[sample.index <24]

sample.plot(kind = 'bar', label = 'AIr bookings < 1 day advance')

plt.legend()

plt.xlabel('Hours between booking and reservation')

plt.subplot(2,2,4)

sample = (hpg_reserve.dateDiff.dt.total_seconds()/3600).value_counts()

sample = sample[sample.index<24]

sample.plot(kind = 'bar', label = 'HPG bookings < 1 day advance')

plt.xlabel('Hours between booking and reservation')

plt.legend()
plt.figure(figsize=(10,10))

m = Basemap(llcrnrlon=120,llcrnrlat=30,urcrnrlon=150,urcrnrlat=46,

            projection='merc')

m.drawcoastlines()

m.shadedrelief()

lon = hpg_store_info.longitude.values

lat = hpg_store_info.latitude.values

xpt,ypt = m(lon,lat)

plt.scatter(xpt, ypt,zorder=10, color = 'yellow', s = 10, 

            alpha = 0.2, label = 'HPG')

lon = air_store_info.longitude.values

lat = air_store_info.latitude.values

xpt,ypt = m(lon,lat)

plt.scatter(xpt, ypt,zorder=10, color = 'red', s = 10,

            alpha = 0.2, label = 'AIR')
plt.figure(figsize=(10,4))

plt.plot_date([min(air_reserve.reserve_datetime),max(air_reserve.reserve_datetime)],[1,1]

              ,'-',label='AIR reserve')

plt.plot_date([min(air_reserve.visit_datetime),max(air_reserve.visit_datetime)],[2,2]

              ,'-',label='AIR visit')

plt.plot_date([min(hpg_reserve.reserve_datetime),max(hpg_reserve.reserve_datetime)],[3,3]

              ,'-',label='HPG reserve')

plt.plot_date([min(hpg_reserve.visit_datetime),max(hpg_reserve.visit_datetime)],[4,4]

              ,'-',label='HPG visit')

plt.plot_date([min(air_visit_data.visit_date),max(air_visit_data.visit_date)],[5,5]

              ,'-',label='AIR total visits')

plt.legend()