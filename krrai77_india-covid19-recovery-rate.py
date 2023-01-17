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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
covid = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
covid1=covid.drop(['Sno','ConfirmedIndianNational','ConfirmedForeignNational'], axis=1)

covid1=covid1[covid1['State/UnionTerritory'] !='Cases being reassigned to states']

covid1.tail()
fig = px.line(covid1, x="Date", y="Confirmed", title='Covid Confirmed Cases',color='State/UnionTerritory')

fig.show()
fig = px.line(covid1, x="Date", y="Cured", title='Covid Cured Cases',color='State/UnionTerritory')

fig.show()
fig = px.line(covid1, x="Date", y="Deaths", title='Covid Deaths',color='State/UnionTerritory')

fig.show()
import plotly.graph_objects as go





# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=covid1['Date'], y=covid1['Confirmed'],

                    mode='lines',

                    name='Confirmed'))

fig.add_trace(go.Scatter(x=covid1['Date'], y=covid1['Cured'],

                    mode='lines+markers',

                    name='Cured'))

fig.add_trace(go.Scatter(x=covid1['Date'], y=covid1['Deaths'],

                    mode='markers', name='Deaths'))



fig.show()
#Calculating Recovery Rate



covid1['Recovery Rate']=round((covid1['Cured']/covid1['Confirmed'])*100)

covid1
from datetime import datetime, timedelta

days_to_subtract = 1

d = datetime.today() - timedelta(days=days_to_subtract)

d = pd.datetime.now().date() - timedelta(days=1)

currdate=d.strftime('%d/%m/%y')
covid2=covid1[covid1['Date'] == currdate]

covidcurr=covid2[['Date', 'State/UnionTerritory', 'Cured', 'Deaths', 'Confirmed','Recovery Rate']].copy()
cm = sns.light_palette("orange", as_cmap=True)

covidcurr.style.background_gradient(cmap=cm)
Maha=covid1[covid1['State/UnionTerritory']=='Maharashtra']

fig = px.line(Maha, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Maharashtra')

fig.show()

Delhi=covid1[covid1['State/UnionTerritory']=='Delhi']

fig = px.line(Delhi, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Delhi')

fig.show()

TN=covid1[covid1['State/UnionTerritory']=='Tamil Nadu']

fig = px.line(TN, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Tamil Nadu')

fig.show()

Karnataka=covid1[covid1['State/UnionTerritory']=='Karnataka']

fig = px.line(Karnataka, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Karnataka')

fig.show()

Kerala=covid1[covid1['State/UnionTerritory']=='Kerala']

fig = px.line(Kerala, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Kerala')

fig.show()

Gujarat=covid1[covid1['State/UnionTerritory']=='Gujarat']

fig = px.line(Gujarat, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Gujarat')

fig.show()

Raja=covid1[covid1['State/UnionTerritory']=='Rajasthan']

fig = px.line(Raja, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Rajasthan')

fig.show()

Tel=covid1[covid1['State/UnionTerritory']=='Telangana']

fig = px.line(Tel, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Telangana')

fig.show()

AP=covid1[covid1['State/UnionTerritory']=='Andhra Pradesh']

fig = px.line(AP, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Andhra Pradesh')

fig.show()

Punjab=covid1[covid1['State/UnionTerritory']=='Punjab']

fig = px.line(Punjab, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Punjab')

fig.show()

WB=covid1[covid1['State/UnionTerritory']=='West Bengal']

fig = px.line(WB, x="Date", y="Recovery Rate", title='Covid Recovery Rate in West Bengal')

fig.show()

Odisha=covid1[covid1['State/UnionTerritory']=='Odisha']

fig = px.line(Odisha, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Odisha')

fig.show()

UP=covid1[covid1['State/UnionTerritory']=='Uttar Pradesh']

fig = px.line(UP, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Uttar Pradesh')

fig.show()

MP=covid1[covid1['State/UnionTerritory']=='Madhya Pradesh']

fig = px.line(MP, x="Date", y="Recovery Rate", title='Covid Recovery Rate in Madhya Pradesh')

fig.show()