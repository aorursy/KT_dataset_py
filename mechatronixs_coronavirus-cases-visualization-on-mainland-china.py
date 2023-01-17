# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go 



import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.shape
data.head()
data1 = data.drop(['SNo', 'Last Update'], axis=1)



data1.head()
data1.info()
uni_dates = list(data1['ObservationDate'].unique())

confirmed=[]

recovered=[]

deaths=[]



for x in uni_dates:

    confirmed.append(data1[data1['ObservationDate']==x].Confirmed.sum())

    recovered.append(data1[data1['ObservationDate']==x].Recovered.sum())

    deaths.append(data1[data1['ObservationDate']==x].Deaths.sum())





line_new = pd.DataFrame()

line_new ['ObservationDate']=uni_dates

line_new['Confirmed']=confirmed

line_new['Recovered']=recovered

line_new['Deaths']=deaths

line_new.tail(10)
fig = go.Figure()

fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 

                         y=line_new['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='Yellow', width=3)))

fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 

                         y=line_new['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=3)))

fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 

                         y=line_new['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=3)))



fig.show()
line_new = line_new.set_index('ObservationDate')

plt.style.use('default') 

plt.figure(figsize=(20,15))

sns.lineplot(data=line_new)

plt.xticks(rotation=15)

plt.title('Number of Coronavirus Cases Over Time', size =20)

plt.xlabel('Time', size=20)

plt.ylabel('Number of Cases', size=20)

plt.show()
plt.figure(figsize=(20,15))

line_new.Confirmed.plot(kind = 'line', color = 'm',label = 'Confirmed',linewidth=2,alpha = 1,linestyle = ':')

line_new.Deaths.plot(kind = 'line', color = 'k',label = 'Deaths',linewidth=3,alpha = 0.7,linestyle = '-.' )

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Date')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
plt.figure(figsize=(20,15))

data3 = data1[data1['Country/Region'] =='Mainland China']

data3.Confirmed.plot(kind = 'line', color = 'y',label = 'Confirmed',linewidth=1,alpha = 1,linestyle = '--')

data3.Recovered.plot(kind = 'line',color = 'c',label = 'Recovered', linewidth = 3, linestyle = '-')

data3.Deaths.plot(kind = 'line', color = 'r',label = 'Deaths',linewidth=2,alpha = 0.7 )

plt.legend(loc='upper right')     # legend = puts label into plot

plt.ylabel('Values',fontsize= 16)

plt.title('Number of Corona Virus Cases in Mainland China',fontsize = 16)            # title = title of plot

plt.legend()

plt.tight_layout()

plt.show()
x = np.array(data3.loc[:,'Confirmed']).reshape(-1,1)

y = np.array(data3.loc[:,'Recovered']).reshape(-1,1)

#Scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y,color='Green')

plt.xlabel('Confirmed')

plt.ylabel('Recovered')

plt.title('Confirmed-Recovered in Mainland China')            # title = title of plot

plt.show()
x = np.array(data3.loc[:,'Confirmed']).reshape(-1,1)

y = np.array(data3.loc[:,'Deaths']).reshape(-1,1)

#Scatter

plt.figure(figsize=[10,10])

plt.plot(x,y,'-',lw=2, color='r')

plt.xlabel('Confirmed')

plt.ylabel('Deaths')

plt.title('Confirmed-Deaths in Mainland China')            # title = title of plot

plt.show()
f,ax1 = plt.subplots(figsize =(30,20))

sns.pointplot(x=data3['Province/State'],y=data3['Confirmed'],color = 'blue')

plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'blue')

plt.ylabel("Confirmed Values",fontsize = 16 , color = 'blue')

plt.title("Confirmed Rate for Every Province/States in Mainland China",fontsize=20)
f,ax1 = plt.subplots(figsize =(30,20))

sns.pointplot(x=data3['Province/State'],y=data3['Deaths'],color = 'black')

plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'red')

plt.ylabel("Deaths",fontsize = 16 , color = 'red')

plt.title("Death Rate for Every Province/States in Mainland China",fontsize=20)
f,ax1 = plt.subplots(figsize =(30,20))

sns.pointplot(x=data3['Province/State'],y=data3['Recovered'],color = 'green')

plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'green')

plt.ylabel("Recovered",fontsize = 16 , color = 'green')

plt.title("Recovered Rate for Every Province/States in Mainland China",fontsize=20)
data7 = data3[data3['Province/State'] =='Hubei']

data7.tail()


f,ax1 = plt.subplots(figsize =(40,30))

sns.barplot(x="Confirmed", y="Recovered",

                  hue="Province/State", data=data7)

plt.xlabel("Total Confirmed Numbers by Time",fontsize = 20 , color = 'green')

plt.ylabel("Total Recovered Numbers by Time",fontsize = 20 , color = 'green')

plt.show()
f,ax1 = plt.subplots(figsize =(35,20))

sns.barplot(x="Confirmed", y="Deaths",

                  hue="Province/State", data=data7,alpha=1,color='red')

plt.xlabel("Total Confirmed Numbers by Time",fontsize = 20 , color = 'red')

plt.ylabel("Total Deaths Numbers by Time",fontsize = 20 , color = 'red')

plt.show()