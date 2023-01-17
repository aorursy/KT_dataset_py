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
import numpy as np # linear algebra

import pandas as pd

train=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

train_old=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train.head()
pd.DataFrame(train[train['Country/Region']=='China']['Date'].groupby(train['Province/State']))

pd.DataFrame(train[train['Country/Region']=='China']['Confirmed'].groupby(train['Date']).sum()).reset_index()

pd.DataFrame(train[train['Country/Region']=='China']['Deaths'].groupby(train['Date']).sum()).reset_index()
data=pd.DataFrame(train[train['Country/Region']=='China'].groupby(train['Date'])['Confirmed','Deaths','Recovered'].sum().sort_values(by='Confirmed')).reset_index()
data.head()    
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

%matplotlib inline
fig1_1=go.Figure()

fig1_1.add_trace(go.Scatter(x=data['Date'],y=data['Confirmed'],mode='lines+markers',marker=dict(size=10,color=1)))
arr=[]

for i in range(0,data.shape[0]):

    if i==0:

        arr.append(data.iloc[0]['Confirmed'])

    else:

        arr.append(data.iloc[i]['Confirmed']-data.iloc[i-1]['Confirmed'])

data['daily_cases']=arr
fig2_1=go.Figure()

fig2_1.add_trace(go.Bar(x=data['Date'],y=data['daily_cases']))
fig3_1=go.Figure()

fig3_1.add_trace(go.Scatter(x=data['Date'],y=data['Deaths'],mode='lines+markers',marker=dict(size=10),marker_color='rgba(152, 0, 0, .8)'))
mortality=[]

for i in range(0,data.shape[0]):

    if i==0:

        mortality.append(data.iloc[0]['Deaths'])

    else:

        mortality.append(data.iloc[i]['Deaths']-data.iloc[i-1]['Deaths'])

data['daily_deaths']=mortality
fig4_1=px.bar(data,x='Date',y='daily_deaths',hover_data=['Date','daily_deaths'])



fig4_1.show()
data['Active']=data['Confirmed']-(data['Deaths'] + data['Recovered'])
fig5_1=go.Figure()

fig5_1.add_trace(go.Scatter(x=data['Date'],y=data['Active'],mode='lines+markers',marker=dict(size=10)))
daily_recovered=[]

for i in range(0,data.shape[0]):

    if i==0:

        daily_recovered.append(data.iloc[0]['Recovered'])

    else:

        daily_recovered.append(data.iloc[i]['Recovered']-data.iloc[i-1]['Recovered'])

data['daily_recovered']=daily_recovered
fig7_1=go.Figure()

fig7_1.add_trace(go.Scatter(x=data['Date'],y=data['daily_cases'],mode='lines+markers',marker=dict(size=10),name='Daily cases'))

fig7_1.add_trace(go.Scatter(x=data['Date'],y=data['daily_recovered'],mode='lines+markers',marker=dict(size=10),name='Daily recovered'))
fig6_1=go.Figure()

fig6_1.add_trace(go.Scatter(x=data['Date'],y=data['Deaths'],mode='lines+markers',marker=dict(size=10,color=1),name='Deaths'))

fig6_1.add_trace(go.Scatter(x=data['Date'],y=data['Recovered'],mode='lines+markers',marker=dict(size=10,color=2),name='Recovered'))
# Outcome of Cases (Recovery rate or Death rate) in China

death_rate=(data['Deaths']/(data['Confirmed']-data['Active']))*100

recover_rate=(data['Recovered']/(data['Confirmed']-data['Active']))*100
fig6_2=go.Figure()

fig6_2.add_trace(go.Scatter(x=data['Date'],y=death_rate,mode='lines+markers',marker=dict(size=10,color=1),name='Deaths'))

fig6_2.add_trace(go.Scatter(x=data['Date'],y=recover_rate,mode='lines+markers',marker=dict(size=10,color=2),name='Recovered'))
# fig, ax = plt.subplots(figsize=(30,10))

# ax.set_yticks([0, 25000,50000, 75000, 100000])

# sns.scatterplot(data['Date'],data['ConfirmedCases'])



# fig2, ax2 = plt.subplots(figsize=(30,10))

# ax2.bar(data['Date'],data['ConfirmedCases'], align="center", width=0.5, alpha=0.5)
# arr=[]

# for i in range(0,data.shape[0]):

#     if i==0:

#         arr.append(data.iloc[0]['ConfirmedCases'])

#     else:

#         arr.append(data.iloc[i]['ConfirmedCases']-data.iloc[i-1]['ConfirmedCases'])

# data['daily_cases']=arr



# fig3, ax3 = plt.subplots(figsize=(30,10))

# ax3.set_yticks([0, 25000,50000, 75000, 100000])

# sns.scatterplot(data['Date'],data['daily_cases'])



# fig4, ax4 = plt.subplots(figsize=(30,10))

# ax4.bar(data['Date'],data['daily_cases'], align="center", width=0.5, alpha=0.5)
# fig6, ax6 = plt.subplots(figsize=(30,10))

# ax6.set_yticks([0,1000,2000,3000,4000])

# ax6.set_title("Fatalities")

# # ax6.set_xticks()

# sns.scatterplot(data['Date'],data['Fatalities'])



# fig8, ax8 = plt.subplots(figsize=(30,10))

# ax8.bar(data['Date'],data['Fatalities'], align="center", width=0.5, alpha=0.5)
# mortality=[]

# for i in range(0,data.shape[0]):

#     if i==0:

#         mortality.append(data.iloc[0]['Fatalities'])

#     else:

#         mortality.append(data.iloc[i]['Fatalities']-data.iloc[i-1]['Fatalities'])

# data['daily_fatalities']=mortality



# fig5, ax5 = plt.subplots(figsize=(30,10))

# sns.barplot(x='Date',y='daily_fatalities',data=data)