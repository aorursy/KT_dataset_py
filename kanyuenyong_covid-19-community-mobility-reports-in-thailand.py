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

import io

import requests









# Path of the file to read, see details at https://www.google.com/covid19/mobility/

# backup url = "https://github.com/sikkha/Covid-19/blob/master/Global_Mobility_Report.csv"



url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}



# real code (denied by maximum try)

s=requests.get(url, headers=headers).content

c=pd.read_csv(io.StringIO(s.decode('utf-8')), dtype={'sub_region_2': str})



mydata = c.loc[c['country_region'] == 'Thailand']



from pandas import DataFrame

import matplotlib.pyplot as plt



#df = DataFrame(mydata,columns=['date','retail_andrecreation_percent_change_from_baseline'])

#df.plot(x ='date', y='retail_andrecreation_percent_change_from_baseline', kind = 'scatter')

#plt.show()



import pandas as pd

from datetime import datetime

import csv

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as ticker





x = mydata['date']

y = mydata['retail_and_recreation_percent_change_from_baseline']

z = mydata['grocery_and_pharmacy_percent_change_from_baseline']

A = mydata['parks_percent_change_from_baseline']

B = mydata['transit_stations_percent_change_from_baseline']

C = mydata['workplaces_percent_change_from_baseline']

D = mydata['residential_percent_change_from_baseline']



#==== multiple plot ===

tick_spacing = 10

fig, ax = plt.subplots(1,1)

#ax.plot(x,y)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Percent change from baseline') 

# plot

plt.plot(x,y)

plt.plot(x,z)

plt.plot(x,A)

plt.plot(x,B)

plt.plot(x,C)

plt.plot(x,D)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()



#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Retail and recreation percent change from baseline') 

plt.plot(x,y)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()



#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Grocery and Pharmacy percent change from baseline') 

plt.plot(x,z)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()



#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Parks percent change from baseline') 

plt.plot(x,A)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()





#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Transit stations percent change from baseline') 

plt.plot(x,B)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()



#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Workplaces percent change from baseline') 

plt.plot(x,C)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()



#====

tick_spacing = 10

fig, ax = plt.subplots(1,1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))



plt.ylabel('Residential percent change from baseline') 

plt.plot(x,D)

# beautify the x-labels

plt.gcf().autofmt_xdate()

plt.show()