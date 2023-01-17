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
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
%matplotlib inline 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
df_git = pd.read_csv(url,na_filter=False)
df_git.head(10)
#Nigeria
ng=df_git.loc[df_git['Country'] == 'Nigeria']
#Ghana
gh=df_git.loc[df_git['Country'] == 'Ghana']
#Tunisia
tu=df_git.loc[df_git['Country'] == 'Tunisia']
#Senegal
sn=df_git.loc[df_git['Country'] == 'Senegal']
gh.tail(10)
df_date=df_git.loc[df_git['Date'] >= '2020-03-01']
df_date.head(10)
df_confirmed = df_date[df_date["Confirmed"]>=1]
df_confirmed = df_confirmed[["Country","Date","Confirmed"]]
df_confirmed.head(10)
#ConfirmedCases Group By Country
ConfirmedCases=df_confirmed.groupby(['Country'])['Confirmed'].sum() 
ConfirmedCases.head(10)
ConfirmedCases.sort_values(ascending=True).plot(x='Country',y='Confirmed',fontsize=18,figsize=(20,100)
                                                ,kind='barh');
west_africa = df_date[ (df_date['Country'] == 'Ghana') | (df_date['Country'] == "Nigeria") |
(df_date['Country'] == "Senegal")| (df_date['Country'] == "Gambia")| (df_date['Country'] == "Benin")|
(df_date['Country'] == "Togo")|(df_date['Country'] == "Burkina Faso")|(df_date['Country'] == "Cote d'Ivoire")|
(df_date['Country'] == "Liberia")|(df_date['Country'] == "Guinea")|(df_date['Country'] == "Niger")|
(df_date['Country'] == "Mali")|(df_date['Country'] == "Guinea-Bissau")|(df_date['Country'] == "Cabo Verde")|
(df_date['Country'] == "Liberia")]
west_africa=west_africa.loc[west_africa['Date'] >= '2020-03-01']
west_africa.head(10)
#data=dt.groupby(["Country/Region"])['ConfirmedCases'].count()
#data.head(50)
#df.groupby('Country/Region').plot(x='Date', y='ConfirmedCases')
group = df_git.groupby('Date')['Deaths'].sum().reset_index()

fig = px.line(group, x="Date", y="Deaths", 
              title="Worldwide Deaths Over Time")

fig.show()
confirmed_total_date = west_africa.groupby(['Date']).agg({'Confirmed':['sum']})
fatalities_total_date = west_africa.groupby(['Date']).agg({'Deaths':['sum']})
total_date = confirmed_total_date.join(fatalities_total_date)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date.plot(ax=ax1)
ax1.set_title("West Africa confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date.plot(ax=ax2, color='orange')
ax2.set_title("West Africa death cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)
group = df_git.groupby('Date')['Deaths'].sum().reset_index()

fig = px.line(group, x="Date", y="Deaths", 
              title="Worldwide Deaths Over Time")

fig.show()
group = df_git.groupby('Date')['Recovered'].sum().reset_index()

fig = px.line(group, x="Date", y="Recovered", 
              title="Worldwide Recovered Over Time")

fig.show()
group = west_africa[west_africa['Country'] == 'Ghana']
group = group.groupby('Date')['Date','Confirmed'].sum().reset_index()
#group = group[group['Country'] == 'Ghana']
fig = px.line(group, x="Date", y="Confirmed", 
              title="Confirmed Cases for Ghana Over Time")

fig.show()
group = west_africa[west_africa['Country'] == 'Senegal']
group = group.groupby('Date')['Date','Confirmed'].sum().reset_index()

fig = px.line(group, x="Date", y="Confirmed", 
              title="Confirmed Cases for Senegal Over Time")

fig.show()
group = west_africa[west_africa['Country'] == 'Nigeria']
group = group.groupby('Date')['Date','Confirmed'].sum().reset_index()

fig = px.line(group, x="Date", y="Confirmed", 
              title="Confirmed Cases for Nigeria Over Time")

fig.show()
#group = west_africa.groupby('Date')['Date','Confirmed','Country'].sum().reset_index()

fig = px.line(west_africa, x="Date", y="Confirmed", color='Country',
              #range_color=[5,8],
              title="Confirmed Cases for West Africa Over Time")

fig.show()
west_africa=west_africa.loc[west_africa['Date'] == '2020-03-29']
fig = px.choropleth(west_africa, locations="Country", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country", range_color=[1,200], 
                    color_continuous_scale='portland', 
                    title='West African Countries with Confirmed Cases', scope='africa', height=800)
#fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.choropleth(west_africa, locations="Country", 
                    locationmode='country names', color="Deaths", 
                    hover_name="Country", range_color=[1,10], 
                    color_continuous_scale='portland',
                    title='Reported Deaths in West Africa', scope='africa', height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.choropleth(west_africa, locations="Country", 
                    locationmode='country names', color="Recovered", 
                    hover_name="Country", range_color=[1,5], 
                    color_continuous_scale='portland',
                    title='Recovered cases in West Africa', scope='africa', height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()