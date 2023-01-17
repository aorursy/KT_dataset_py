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
def india_data_confirmed():

    url_confirmed = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv"

    confirmed  = pd.read_csv(url_confirmed)

    confirmed = confirmed[confirmed['Country/Region']=='India'].groupby('Country/Region').sum().drop(["Lat","Long"],axis = 1).T.reset_index()

    confirmed.rename(columns = {'index':'Date','India':'Confirmed Cases'},inplace = True)

    confirmed.columns.name = ""

    confirmed['Date'] = pd.to_datetime(confirmed['Date'])

    confirmed.sort_values('Date')

    return confirmed



data_confirmed = india_data_confirmed()

data_confirmed
def india_data_recovered():

    url_recovered = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv"

    recovered  = pd.read_csv(url_recovered)

    recovered = recovered[recovered['Country/Region']=='India'].groupby('Country/Region').sum().drop(["Lat","Long"],axis = 1).T.reset_index()

    recovered.rename(columns = {'index':'Date','India':'Confirmed Cases'},inplace = True)

    recovered.columns.name = ""

    recovered['Date'] = pd.to_datetime(recovered['Date'])

    recovered.sort_values('Date')

    return recovered



data_recovered = india_data_recovered()

data_recovered
def india_data_death():

    url_death = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv"

    death  = pd.read_csv(url_death)

    death = death[death['Country/Region']=='India'].groupby('Country/Region').sum().drop(["Lat","Long"],axis = 1).T.reset_index()

    death.rename(columns = {'index':'Date','India':'Confirmed Cases'},inplace = True)

    death.columns.name = ""

    death['Date'] = pd.to_datetime(death['Date'])

    death.sort_values('Date')

    return death



data_death = india_data_death()

data_death
def full_data():

    data_confirmed = india_data_confirmed()

    data_recovered = india_data_recovered()

    data_death = india_data_death()

    full = pd.merge(data_confirmed,data_recovered, on='Date')

    full = pd.merge(full,data_death, on='Date')

    full.rename(columns = {"Confirmed Cases_x":"Confirmed Cases","Confirmed Cases_y":"Recovered Cases","Confirmed Cases":"Death Cases"},inplace = True)

    #full = pd.concat([data_confirmed,data_recovered,data_death],axis = 1)

    full.set_index('Date',inplace = True)

    return full



india_data = full_data()

india_data

    
import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.dates as mdates
#plot data

fig1, ax1 = plt.subplots(figsize=(15,7))

ax1.bar(india_data.index,india_data['Confirmed Cases'],color = 'purple')

ax1.xaxis.set_major_locator(mdates.WeekdayLocator())

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax1.set_title("Confirmed Cases in India",size = 30)

ax1.set_xlabel("Dates",size = 15,weight = 'bold')

ax1.set_ylabel("Number of cases",size = 15,weight = 'bold')

ax1.annotate(india_data.loc['2020-04-25']['Confirmed Cases'],xy = ('2020-04-22',india_data.loc['2020-04-25']['Confirmed Cases']+200),size = 15)
#plot data

fig2, ax2 = plt.subplots(figsize=(15,7))

ax2.bar(india_data.index,india_data['Recovered Cases'],color = 'green')

ax2.xaxis.set_major_locator(mdates.WeekdayLocator())

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax2.set_title("Recovered Cases in India",size = 30)

ax2.set_xlabel("Dates",size = 15,weight = 'bold')

ax2.set_ylabel("Number of recovery",size = 15,weight = 'bold')

ax2.annotate(india_data.loc['2020-04-25']['Recovered Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Recovered Cases']+50),size = 15)
#plot data

fig3, ax3 = plt.subplots(figsize=(15,7))

ax3.bar(india_data.index,india_data['Death Cases'],color = 'red')

ax3.xaxis.set_major_locator(mdates.WeekdayLocator())

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax3.set_title("Death Cases in India",size = 30)

ax3.set_xlabel("Dates",size = 15,weight = 'bold')

ax3.set_ylabel("Number of cases",size = 15,weight = 'bold')

ax3.annotate(india_data.loc['2020-04-25']['Death Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Death Cases']+10),size = 15)
fig4, ax4 = plt.subplots(figsize=(15,7))

ax4.plot(india_data.index,india_data['Confirmed Cases'],color = 'purple',lw = 3,alpha = 0.6)

ax4.plot(india_data.index,india_data['Recovered Cases'],color = 'green',lw = 3,alpha = 0.6)

ax4.plot(india_data.index,india_data['Death Cases'],color = 'red',lw = 3,alpha = 0.6)

ax4.xaxis.set_major_locator(mdates.WeekdayLocator())

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax4.set_title("COVID-19 Data of India",size = 30)

ax4.set_xlabel("Dates",size = 15,weight = 'bold')



ax4.annotate(india_data.loc['2020-04-25']['Confirmed Cases'],xy = ('2020-04-22',india_data.loc['2020-04-25']['Confirmed Cases']+400),size = 15)

ax4.annotate(india_data.loc['2020-04-25']['Recovered Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Recovered Cases']+400),size = 15)

ax4.annotate(india_data.loc['2020-04-25']['Death Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Death Cases']+400),size = 15)
fig5, ax5 = plt.subplots(figsize=(15,7))

ax5.plot(india_data.index,india_data['Confirmed Cases'],color = 'purple',lw = 3,alpha = 0.6)

ax5.fill_between(india_data.index,india_data['Confirmed Cases'],color = 'purple',lw = 3,alpha = 0.2)

ax5.plot(india_data.index,india_data['Recovered Cases'],color = 'green',lw = 3,alpha = 0.6)

ax5.plot(india_data.index,india_data['Death Cases'],color = 'red',lw = 3,alpha = 0.6)

ax5.fill_between(india_data.index,india_data['Recovered Cases'],color = 'green',lw = 3,alpha = 0.6)

ax5.fill_between(india_data.index,india_data['Death Cases'],color = 'red',lw = 3,alpha = 0.6)

ax5.xaxis.set_major_locator(mdates.WeekdayLocator())

ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax5.set_title("COVID-19 Data of India",size = 30)

ax5.set_xlabel("Dates",size = 15,weight = 'bold')



ax5.annotate(india_data.loc['2020-04-25']['Confirmed Cases'],xy = ('2020-04-22',india_data.loc['2020-04-25']['Confirmed Cases']+400),size = 15)

ax5.annotate(india_data.loc['2020-04-25']['Recovered Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Recovered Cases']+400),size = 15)

ax5.annotate(india_data.loc['2020-04-25']['Death Cases'],xy = ('2020-04-23',india_data.loc['2020-04-25']['Death Cases']+400),size = 15)
labels = ['Confirmed','Recovered','Death']

sizes = list(india_data.loc['2020-04-25'])

explode = (0.05,0.05,0.05)  

colours = ['#66b3ff','#99ff99','#ff9999']



fig6, ax6 = plt.subplots(figsize=(7,7))

patches, texts, autotexts = ax6.pie(sizes, explode=explode, labels=labels, pctdistance=0.75,autopct='%1.1f%%',shadow=False, startangle=0,colors = colours)



for text in texts:

    text.set_size(20)

for autotext in autotexts:

    autotext.set_size(15)

    

centre_circle = plt.Circle((0,0),0.60,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



ax6.set_title("%age wise Data of COVID-19 in India",size = 25,color = 'magenta')

plt.tight_layout()



plt.show()