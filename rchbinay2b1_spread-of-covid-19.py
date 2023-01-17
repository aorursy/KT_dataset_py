# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('..//input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
mydata=pd.read_csv('../input/full_data.csv')

# See first few lines of the data#

mydata.head()
#Let us convert date field to datetime

mydata.date=pd.to_datetime(mydata.date)
#Make this column as index column of mydata

mydata.set_index(mydata.date,inplace=True)
#Let us try to find out which country has maximum number of total cases

mydata_groupby_country = mydata.groupby('location').sum()
#sorting the result in descending order and getting first highest 11 results of new_cases

mydata_groupby_country.sort_values('new_cases',ascending=False).head(11)
# we will concentrate on these top countries so let us filter the data for these countries by creating a variable for criteria

crit1 =mydata.location=='China'

crit2 =mydata.location=='Italy'

crit3 =mydata.location=='United States'

crit4 =mydata.location=='Spain'

crit5 =mydata.location=='Germany'

crit6 =mydata.location=='Iran'

crit7 =mydata.location=='France'

crit8 =mydata.location=='South Korea'

crit9 =mydata.location=='Switzerland'

crit10 =mydata.location=='United Kingdom'
#Let us frame separate dataframes for each of these countries

mydata_china= mydata[crit1]

mydata_italy= mydata[crit2]

mydata_us= mydata[crit3]

mydata_spain= mydata[crit4]

mydata_germany= mydata[crit5]

mydata_iran= mydata[crit6]

mydata_france= mydata[crit7]

mydata_southk= mydata[crit8]

mydata_switz= mydata[crit9]

mydata_uk= mydata[crit10]
# Let us try to see weekly progression of spread of the desease

mydata_china_weekly=mydata_china['new_cases'].resample('W',how=sum)

mydata_italy_weekly=mydata_italy['new_cases'].resample('W',how=sum)

mydata_us_weekly=mydata_us['new_cases'].resample('W',how=sum)

mydata_spain_weekly= mydata_spain['new_cases'].resample('W',how=sum)

mydata_germany_weekly= mydata_germany['new_cases'].resample('W',how=sum)

mydata_iran_weekly= mydata_iran['new_cases'].resample('W',how=sum)

mydata_france_weekly= mydata_france['new_cases'].resample('W',how=sum)

mydata_southK_weekly= mydata_southk['new_cases'].resample('W',how=sum)

mydata_switz_weekly= mydata_switz['new_cases'].resample('W',how=sum)

mydata_uk_weekly= mydata_uk['new_cases'].resample('W',how=sum)
#Let us plot the result

fig = plt.figure(figsize=(15,15))

ax1=fig.add_subplot(2,2,1)

ax1=mydata_china_weekly.plot(kind='bar',subplots=True,label='CHINA',figsize=(12,4))



ax2=fig.add_subplot(2,2,2)

ax2=mydata_italy_weekly.plot(kind='bar',subplots=True,label='ITALY')



ax3=fig.add_subplot(2,2,3)

ax3=mydata_us_weekly.plot(kind='bar',subplots=True,label='UNITED STATE')



ax4=fig.add_subplot(2,2,4)

ax4=mydata_spain_weekly.plot(kind='bar',subplots=True,label='SPAIN')

fig = plt.figure(figsize=(12,12))

ax5=fig.add_subplot(2,2,1)

ax5=mydata_germany_weekly.plot(kind='bar',subplots=True,label='GERMANY')



ax6=fig.add_subplot(2,2,2)

ax6=mydata_iran_weekly.plot(kind='bar',subplots=True,label='IRAN')



ax7=fig.add_subplot(2,2,3)

ax7=mydata_france_weekly.plot(kind='bar',subplots=True,label='France')



ax8=fig.add_subplot(2,2,4)

ax5=mydata_southK_weekly.plot(kind='bar',subplots=True,label='SOUTH KOREA')

fig = plt.figure(figsize=(12,12))

ax5=fig.add_subplot(2,2,1)

ax5=mydata_switz_weekly.plot(kind='bar',subplots=True,label='SWITZERLAND')



ax6=fig.add_subplot(2,2,2)

ax6=mydata_uk_weekly.plot(kind='bar',subplots=True,label='UK')

crit11 =mydata.location=='India'
mydata_india=mydata[crit11]
mydata_india_weekly= mydata_india['new_cases'].resample('W',how=sum)
fig = plt.figure(figsize=(12,12))

ax_ind=fig.add_subplot(2,2,1)

ax_ind=mydata_india_weekly.plot(kind='bar',subplots=True,label='INDIA')