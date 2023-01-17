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
import pandas as pd

import matplotlib.pyplot as plt

ds = pd.read_csv('../input/ardd_fatal_crashes_march.csv', index_col=['CrashID']) 

ds.head(10)
col = ds.columns

print(col)
# Get each year total fatals for Crashes

filteredData = ds.Year == 1989

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1990

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1991

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1992

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1993

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1994

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1995

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1996

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1997

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1998

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 1999

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2000

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2001

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2002

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2003

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2004

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2005

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2006

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2007

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2008

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2009

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2010

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2011

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2012

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2013

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2014

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2015

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2016

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2017

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2018

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2019

print((ds[filteredData])['NumberFatalities'].sum())

filteredData = ds.Year == 2020

print((ds[filteredData])['NumberFatalities'].sum())
# Yearly Fatals per 10,000 People - did not consider

data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"], 

        'tot # of Accidents':[2800,2331,2113,1974,1953,1928,2017,1970,1767,1755,1764,1817,

                              1737,1715,1621,1583,1627,1598,1603,1437,1491,1353,

                              1277,1300,1187,1151,1204,1292,1221,1135,1194,181],

        'tot_population':[16936723,17169768,17378981,17557133,17719090,17893433,18119616,18330079,18510004,18705620,18919210,

                19141036,19386461,19605441,19827155,20046003,20311543,20627547,21016121,21475625,21865623,22172469,

                22522197,22928023,23297777,23640331,23984581,24389684,24773350,25171439,25464116,25444328]}

new_ds1 = pd.DataFrame.from_dict(data)

# print(new_ds)



new_ds1['per 10000 ppl'] = ((new_ds1['tot # of Accidents'] / new_ds1['tot_population']) * 10000).round(2)

print(new_ds1)



new_ds1.plot(kind='bar', x='Year',y='per 10000 ppl',ylim =(0.0,1.6), color='green')

plt.title("Yearly Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Get yearly tot # of Crashes for Graph 01

ds.Year.value_counts()
# Graph 01- Yearly Fatal Crashes per 10,000 People



data = {'Year': ["1989", "1990", "1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",

                "2001", "2002", "2003", "2004","2005","2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"], 

        'tot # of Accidents':[2407,2050,1874,1736,1737,1702,1822,1768,1601,1573,1553,1628,

                              1584,1525,1445,1444,1472,1452,1453,1315,1347,1233,

                              1151,1190,1101,1051,1100,1198,1125,1055,1108,159],

        'tot_population':[16936723,17169768,17378981,17557133,17719090,17893433,18119616,18330079,18510004,18705620,18919210,

                19141036,19386461,19605441,19827155,20046003,20311543,20627547,21016121,21475625,21865623,22172469,

                22522197,22928023,23297777,23640331,23984581,24389684,24773350,25171439,25464116,25444328]}



Yearly_Crashes_ds = pd.DataFrame.from_dict(data)

# print(new_ds)



Yearly_Crashes_ds['per 10000 ppl'] = ((Yearly_Crashes_ds['tot # of Accidents'] / Yearly_Crashes_ds['tot_population']) * 10000).round(2)

print(Yearly_Crashes_ds)



Yearly_Crashes_ds.plot(kind='bar', x='Year',y='per 10000 ppl',ylim =(0.0,1.6), color='green')

plt.title("Yearly Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Get 1989 to 2020 total # of Crashes in each state for Graph 02

ds['State'].value_counts()
# Graph 02- Each State # of Fatal Crashes per 10,000 People - from 1989-2020



data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[14260,10161,9081,5459,3918,1336,1383,424],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}



state_crashes_ds = pd.DataFrame.from_dict(data)

# print(state_crashes_ds)



state_crashes_ds['per 10000 ppl'] = ((state_crashes_ds['tot # of Accidents'] / state_crashes_ds['tot_population']) * 10000).round()

print(state_crashes_ds)



state_crashes_ds.plot(kind='bar', x='State',y='per 10000 ppl', color='pink', ylim=(0,60))

plt.title("Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Get 1989 to 2020 total # of Crashes based on speed limit for Graph 03

ds['SpeedLimit'].value_counts()
# Graph 03- % of Fatal Crashes based on Speed Limit for 1989- 2020



data = {'SpeedLimit': ["<=40","41-50","51-60","61-70","71-80","81-90","91-100","101-110",">110"], 

        'tot # of Accidents':[373,2598,12698,2493,5351,971,15194,4962,87]}



speed_crash_ds = pd.DataFrame.from_dict(data)

# print(speed_crash_ds)



speed_crash_ds['Crash %'] = ((speed_crash_ds['tot # of Accidents'] / speed_crash_ds['tot # of Accidents'].sum()) * 100).round(2)

print(speed_crash_ds)



speed_crash_ds.plot(kind='bar', x='SpeedLimit',y='Crash %', color='brown', ylim=(0,40))

plt.title("% of Fatal Crashes based on Speed Limit", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Get only 2019 total # of Crashes in each state for Graph 04

filteredData = ds.Year == 2019

(ds[filteredData])['State'].value_counts()
# Graph 04 - Year 2019 Each State Fatal Crashes per 10,000 People



data = {'State': ["NSW", "VIC", "QLD","WA","SA","TAS","NT","ACT"], 

        'tot # of Accidents':[331,252,195,154,110,32,28,6],

        'tot_population':[8117976,6629870,5115451,2630557,1756494,535500,245562,428060]}

crashes2019_ds = pd.DataFrame.from_dict(data)

# print(crashes2019_ds)



crashes2019_ds['per 10000 ppl'] = ((crashes2019_ds['tot # of Accidents'] / crashes2019_ds['tot_population']) * 10000).round(2)

print(crashes2019_ds)



crashes2019_ds.plot(kind='bar', x='State',y='per 10000 ppl', color='green', ylim=(0.0,1.6))

plt.title("2019 Each State Fatal Crashes per 10,000 People", y=1.05);

plt.ylabel("# of Fatal Crashes per 10,000 People", labelpad=14)
# Get total # of Crashes in daytime and nighttime for Graph 05

ds.TimeofDay.value_counts()
# Graph 05 - Percentage of Fatal Crashes based on Time of the Day



data = {'Time of Day': ["Day", "Night"], 'count':[26143, 19816]}

time_base_ds = pd.DataFrame.from_dict(data)

# print(time_base_ds)



time_base_ds['Percentage Value'] = ((time_base_ds['count'] / time_base_ds['count'].sum()) * 100).round(2)

print(time_base_ds)



time_base_ds.plot(kind='bar',ylim =(0,100), x='Time of Day',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Time of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Get total # of Crashes happen in nighttime  base on days of week for Graph 06



night_time_data = ds.TimeofDay =='Night'

(ds[night_time_data])['DayWeek'].value_counts()
# Graph 06 - Percentage of Fatal Crashes based on Night Time of Days of week



data = {'Days of Week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 'count':[1877, 1900, 2176, 2599, 3526, 4292, 3446]}

night_time_ds = pd.DataFrame.from_dict(data)

# print(night_time_ds)



night_time_ds['Percentage Value'] = ((night_time_ds['count'] / night_time_ds['count'].sum()) * 100).round(2)

print(night_time_ds)



night_time_ds.plot(kind='bar',ylim =(0,25), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Night Time of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Nighttime", labelpad=14)
# Get total # of Crashes happen in daytime base on days of week for Graph 07



day_time_data = ds.TimeofDay =='Day'

(ds[day_time_data])['DayWeek'].value_counts()
# Graph 07 - Percentage of Fatal Crashes based on Night Time of Days of week



data = {'Days of Week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 'count':[3497, 3513, 3625, 3672, 3986, 4057, 3793]}

day_time_ds = pd.DataFrame.from_dict(data)

# print(day_time_ds)



day_time_ds['Percentage Value'] = ((day_time_ds['count'] / day_time_ds['count'].sum()) * 100).round(2)

print(day_time_ds)



day_time_ds.plot(kind='bar',ylim =(0,20), x='Days of Week',y='Percentage Value', color='green')

plt.title("Percentage of Fatal Crashes based on Daytime of the Day ", y=1.05);

plt.ylabel("% of Fatal Crashes in Daytime", labelpad=14)
# Get total # of Crashes happen in Christmas base on years for Graph 08



Christmas_Data = ds.ChristmasPeriod == 'Yes'

(ds[Christmas_Data])['Year'].value_counts()
# Graph 08 - % of Fatal Crashes in Christmas Period 2006 - 2019



data = {'Year': ["2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[50,32,38,37,27,

                              46,46,32,22,35,34,40,43,24]}



christmas_ds = pd.DataFrame.from_dict(data)

# print(christmas_ds)



christmas_ds['Crash %'] = ((christmas_ds['tot # of Accidents'] / christmas_ds['tot # of Accidents'].sum()) * 100).round(2)

print(christmas_ds)



christmas_ds.plot(kind='bar', x='Year',y='Crash %', color='purple', ylim=(0,12))

plt.title("% of Fatal Crashes in Christmas Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Get total # of Crashes happen in Easter base on years for Graph 09



Easter_Data = ds.EasterPeriod == 'Yes'

(ds[Easter_Data])['Year'].value_counts()
# Graph 09 - % of Fatal Crashes in Easter Period 2006 - 2019



data = {'Year': ["2006", "2007","2008","2009","2010",

                "2011","2012","2013","2014","2015","2016","2017","2018","2019"], 

        'tot # of Accidents':[18,25,14,18,12,

                              20,10,19,12,19,8,11,14,18]}



easter_ds = pd.DataFrame.from_dict(data)

# print(easter_ds)



easter_ds['Crash %'] = ((easter_ds['tot # of Accidents'] / easter_ds['tot # of Accidents'].sum()) * 100).round(2)

print(easter_ds)



easter_ds.plot(kind='bar', x='Year',y='Crash %', color='orange', ylim=(0,12))

plt.title("% of Fatal Crashes in Easter Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)
# Graph 10 - % of Fatal crashes Comparison of Christmas & Easter Period 2006 - 2019



import matplotlib.pyplot as plt



Christmas = [9.88,6.32,7.51,7.31,5.34,9.09,9.09,6.32,4.35,6.92, 6.72,7.91,8.50,4.74]

Easter = [8.26,11.47,6.42,8.26,5.50,9.17 ,4.59 ,8.72,5.50,8.72,3.67,5.05,6.42,8.26]

index = ["2006", "2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"]

df = pd.DataFrame({'Christmas Period': Christmas,

                   'Easter Period': Easter}, index=index)

ax = df.plot.bar(rot=90)





# new_ds.plot(kind='bar', x='Year',y='Crash %', color='orange', ylim=(0,12))

plt.title("Fatal crashes Comparison of Christmas & Easter Period", y=1.05);

plt.ylabel("% of Fatal Crashes", labelpad=14)

plt.xlabel("Year", labelpad=14)