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
# Loading and visualizing training data
training_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
training_data
# Countries
print('Countries:',training_data.Country_Region.unique())
print('Number of countries:',len(training_data.Country_Region.unique()))
# Time Frame
print('Data has been collected from',training_data.Date.min(),'to',training_data.Date.max())
# Trend of confirmed case and fatality growth in Germany
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

germany_data = training_data.loc[training_data['Country_Region'] == 'Germany']
print(germany_data)
print('Number of province states:',len(germany_data.Province_State.unique()),'(',germany_data.Province_State.unique(),')')
fig, (ax1, ax2) = plt.subplots(2)
ax1.title.set_text('Confirmed cases')
ax2.title.set_text('Fatalities')
ax2.xaxis.set_major_locator(MultipleLocator(15))
germany_data_specific = germany_data[['Id','Date','ConfirmedCases','Fatalities']]
plt.xticks(rotation=90)
ax1.plot(germany_data_specific['Date'],germany_data_specific['ConfirmedCases'])
ax1.set_xticks([])
ax2.plot(germany_data_specific['Date'],germany_data_specific['Fatalities'])
fig.text(0.5, 0.04, 'Date', ha='center')
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical')
# Trend of confirmed case and fatality growth in China
import math
from datetime import datetime
import matplotlib.pyplot as plt

china_data = training_data.loc[training_data['Country_Region'] == 'China']
# print(china_data.to_string())
print('Number of province states:',len(china_data.Province_State.unique()),'(',china_data.Province_State.unique(),')')
tibet_data = china_data.loc[china_data['Province_State'] == 'Tibet']
# print(tibet_data.to_string())
# Plotting trend for each province
for a in china_data.Province_State.unique():
    plt.figure()
    china_province_specific = china_data.loc[china_data['Province_State'] == a]
    plt.plot(china_province_specific['Date'],china_province_specific['ConfirmedCases'],label = 'Confirmed cases')
    plt.plot(china_province_specific['Date'],china_province_specific['Fatalities'],label = 'Fatalities')
    plt.suptitle(a)
    ax = plt.axes()
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()
# Comparing different provinces in China
X = np.arange(len(china_data.Province_State.unique()))
fig = plt.figure()
width = 0.35 
max_confirmed_province=[]
max_deaths_province=[]
for a in china_data.Province_State.unique():
    data = china_data.loc[china_data['Province_State'] == a]
    max_confirmed_province.append(data['ConfirmedCases'].max())
    max_deaths_province.append(data['Fatalities'].max())
plt.bar(X, max_confirmed_province, width,label='Confirmed cases')
plt.bar(X + width, max_deaths_province, width,label='Fatalities')
ax1 = plt.axes()
ax1.set_xlabel('All Chinese provinces')
ax1.set_ylabel('Count')
ax1.yaxis.set_major_locator(MultipleLocator(5000))
plt.xticks(rotation='vertical')
plt.xticks(X + width / 2, china_data.Province_State.unique())    
plt.legend()
plt.show()
# Comparing different provinces in China except Hubei
X = np.arange(len(china_data.Province_State.unique())-1)
fig = plt.figure()
width = 0.35 
max_confirmed_province=[]
max_deaths_province=[]
for a in china_data.Province_State.unique():
    data = china_data.loc[china_data['Province_State'] == a]
    if a!='Hubei':
        max_confirmed_province.append(data['ConfirmedCases'].max())
        max_deaths_province.append(data['Fatalities'].max())
plt.bar(X, max_confirmed_province, width,label='Confirmed cases')
plt.bar(X + width, max_deaths_province, width,label='Fatalities')
ax1 = plt.axes()
ax1.set_xlabel('All Chinese provinces (except Hubei)')
ax1.set_ylabel('Count')
ax1.yaxis.set_major_locator(MultipleLocator(100))
plt.xticks(rotation='vertical')
temp = china_data.Province_State.unique()
result = np.where(temp == 'Hubei')
provinces_except_Hubei = np.delete(temp,result)
plt.xticks(X + width / 2, provinces_except_Hubei)    
plt.legend()
plt.show()
# Comparing death counts in the different provinces of China except Hubei   
X = np.arange(len(china_data.Province_State.unique())-1)
fig = plt.figure()
width = 0.35 
max_deaths_province=[]
for a in china_data.Province_State.unique():
    data = china_data.loc[china_data['Province_State'] == a]
    if a!='Hubei':
        max_deaths_province.append(data['Fatalities'].max())

plt.bar(X + width, max_deaths_province, width, color='darkorange')
ax1 = plt.axes()
ax1.title.set_text('Fatalities')
ax1.set_xlabel('Province')
ax1.set_ylabel('Count')
ax1.yaxis.set_major_locator(MultipleLocator(5))
# Displays count above each bar
# for i, v in enumerate(max_deaths_province):
    # ax1.text(i , v + 0.2, str(int(v)), color='blue', fontweight='bold', rotation='vertical')
plt.xticks(rotation='vertical')
temp = china_data.Province_State.unique()
result = np.where(temp == 'Hubei')
provinces_except_Hubei = np.delete(temp,result)
plt.xticks(X + width / 2, provinces_except_Hubei)    
plt.show()    
# Trend of confirmed case and fatality growth in India
import math
from datetime import datetime
import matplotlib.pyplot as plt

india_data = training_data.loc[training_data['Country_Region'] == 'India']
# print(india_data.to_string())
print('Number of province states:',len(india_data.Province_State.unique()),'(',india_data.Province_State.unique(),')')
fig, (ax1, ax2) = plt.subplots(2)
ax1.title.set_text('Confirmed cases in India')
ax2.title.set_text('Fatalities in India')
ax2.xaxis.set_major_locator(MultipleLocator(15))
plt.xticks(rotation=90)
ax1.plot(india_data['Date'],india_data['ConfirmedCases'])
ax1.set_xticks([])
ax2.plot(india_data['Date'],india_data['Fatalities'])
fig.text(0.5, 0.04, 'Date', ha='center')
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical')
# Trend of confirmed case and fatality growth in Italy
import math
from datetime import datetime
import matplotlib.pyplot as plt

italy_data = training_data.loc[training_data['Country_Region'] == 'Italy']
# print(italy_data.to_string())
print('Number of province states:',len(italy_data.Province_State.unique()),'(',italy_data.Province_State.unique(),')')
fig, (ax1, ax2) = plt.subplots(2)
ax1.title.set_text('Confirmed cases in Italy')
ax2.title.set_text('Fatalities in Italy')
ax2.xaxis.set_major_locator(MultipleLocator(15))
ax1.yaxis.set_major_locator(MultipleLocator(15000))
ax2.yaxis.set_major_locator(MultipleLocator(2000))
plt.xticks(rotation=90)
ax1.plot(italy_data['Date'],italy_data['ConfirmedCases'])
ax1.set_xticks([])
ax2.plot(italy_data['Date'],italy_data['Fatalities'])
fig.text(0.5, 0.04, 'Date', ha='center')
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical')
# Trend of confirmed case and fatality growth in Spain
import math
from datetime import datetime
import matplotlib.pyplot as plt

spain_data = training_data.loc[training_data['Country_Region'] == 'Spain']
# print(spain_data.to_string())
print('Number of province states:',len(spain_data.Province_State.unique()),'(',spain_data.Province_State.unique(),')')
fig, (ax1, ax2) = plt.subplots(2)
ax1.title.set_text('Confirmed cases in Spain')
ax2.title.set_text('Fatalities in Spain')
ax2.xaxis.set_major_locator(MultipleLocator(15))
ax1.yaxis.set_major_locator(MultipleLocator(15000))
ax2.yaxis.set_major_locator(MultipleLocator(2000))
plt.xticks(rotation=90)
ax1.plot(spain_data['Date'],spain_data['ConfirmedCases'])
ax1.set_xticks([])
ax2.plot(spain_data['Date'],spain_data['Fatalities'])
fig.text(0.5, 0.04, 'Date', ha='center')
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical')
# Trend of confirmed case and fatality growth in the United Kingdom
import math
from datetime import datetime
import matplotlib.pyplot as plt

uk_data = training_data.loc[training_data['Country_Region'] == 'United Kingdom']
# print(uk_data.to_string())
uk_data = uk_data.fillna('Unknown province')
# print(uk_data.to_string())
print('Number of province states:',len(uk_data.Province_State.unique()),'(',uk_data.Province_State.unique(),')')
for a in uk_data.Province_State.unique():
    plt.figure()
    uk_province_specific = uk_data.loc[uk_data['Province_State'] == a]
    plt.plot(uk_province_specific['Date'],uk_province_specific['ConfirmedCases'],label = 'Confirmed cases')
    plt.plot(uk_province_specific['Date'],uk_province_specific['Fatalities'],label = 'Fatalities')
    plt.suptitle(a)
    ax = plt.axes()
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()
# Trend of confirmed case and fatality growth in Indonesia
import math
from datetime import datetime
import matplotlib.pyplot as plt

indonesia_data = training_data.loc[training_data['Country_Region'] == 'Indonesia']
# print(indonesia_data.to_string())
print('Number of province states:',len(indonesia_data.Province_State.unique()),'(',indonesia_data.Province_State.unique(),')')
fig, (ax1, ax2) = plt.subplots(2)
ax1.title.set_text('Confirmed cases in Indonesia')
ax2.title.set_text('Fatalities in Indonesia')
ax2.xaxis.set_major_locator(MultipleLocator(15))
ax1.yaxis.set_major_locator(MultipleLocator(500))
ax2.yaxis.set_major_locator(MultipleLocator(50))
plt.xticks(rotation=90)
ax1.plot(indonesia_data['Date'],indonesia_data['ConfirmedCases'])
ax1.set_xticks([])
ax2.plot(indonesia_data['Date'],indonesia_data['Fatalities'])
fig.text(0.5, 0.04, 'Date', ha='center')
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical')