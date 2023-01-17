# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.graph_objects as go 

import seaborn as sns

import plotly.express as px

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_all = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data_all.head()
data_all.rename(columns={'Province/State':'State'}, inplace = True )

data_all.rename(columns={'Country/Region':'Country'}, inplace = True )

data_all.rename(columns={'ObservationDate':'Date'}, inplace = True )



data_all['Country'].replace(['Mainland China'] ,['China'],inplace = True)

# Create derived column called Active Cases 

data_all['Active'] = data_all['Confirmed'] - data_all['Recovered'] - data_all['Deaths']

# Convert date column to appropriate data type

data_all['Date'] = pd.to_datetime(data_all['Date'])
data_all.head()
temp = data_all.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp.style.background_gradient(cmap='Set1')

temp.tail()
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp['Date'], 

                         y=temp['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='Yellow', width=3)))

fig.add_trace(go.Scatter(x=temp['Date'], 

                         y=temp['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=3)))

fig.add_trace(go.Scatter(x=temp['Date'], 

                         y=temp['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=3)))



fig.add_trace(go.Scatter(x=temp['Date'], 

                         y=temp['Active'],

                         mode='lines+markers',

                         name='Actived',

                         line=dict(color='Grey', width=3)))

fig.show()
x = np.array(temp.loc[:,'Confirmed']).reshape(-1,1)

y = np.array(temp.loc[:,'Active']).reshape(-1,1)

#Scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y,color='Red')

plt.xlabel('Confirmed')

plt.ylabel('Active')

plt.title('Confirmed-Active in World')            # title = title of plot

plt.show()
plt.figure(figsize=(20,20))

top_date = data_all[data_all['Date'] == data_all['Date'].max()]

g = sns.PairGrid(top_date, vars=['Confirmed', 'Deaths', 'Recovered', 'Active'],palette="husl")

g.map(plt.scatter, alpha=0.8)

g.add_legend();
top_cntry_con = top_date.groupby(by = 'Country')['Confirmed'].sum().sort_values(ascending = False).head(20).reset_index()

plt.figure(figsize=(20,15))

sns.barplot(x=top_cntry_con['Country'], y=top_cntry_con['Confirmed'])

plt.xticks(rotation= 45)

plt.xlabel('Countries')

plt.ylabel('Confirmed')

plt.title('Country-Confirmed Quantities')
top_cntry_con
top_cntry_ac = top_date.groupby(by = 'Country')['Active'].sum().sort_values(ascending = False).head(20).reset_index()

plt.figure(figsize=(20,15))

sns.barplot(x=top_cntry_ac['Country'], y=top_cntry_ac['Active'],palette="Set2")

plt.xticks(rotation= 45)

plt.xlabel('Countries')

plt.ylabel('Active')

plt.title('Country-Active Quantities')
f, ax = plt.subplots(figsize=(20, 15))

top_cntry_dth = top_date.groupby(by = 'Country')['Deaths'].sum().sort_values(ascending = False).head(20).reset_index()

sns.barplot(x="Country", y="Deaths", data=top_cntry_dth, palette="GnBu_d")

plt.xlabel('Countries')

plt.ylabel('Deaths')

plt.title('Country-Deaths Graphics')

plt.show()

top_cntry_dth
f, ax = plt.subplots(figsize=(20, 15))

top_cntry_rec = top_date.groupby(by = 'Country')['Recovered'].sum().sort_values(ascending = False).head(20).reset_index()

sns.set_color_codes("pastel")

sns.barplot(x="Country", y="Recovered", data=top_cntry_rec, palette="BuGn_r")

plt.xticks(rotation= 30)

plt.xlabel('Countries')

plt.ylabel('Recovered')

plt.title('Country-Recovered Graphics')

plt.show()
top_cntry_con.rename(columns={'Confirmed':'Total Confirmed'}, inplace=True)

top_con_rec = top_date.groupby(by = 'Recovered')['Confirmed'].sum().sort_values(ascending = False).head(20).reset_index()

top_con_act = top_date.groupby(by = 'Active')['Confirmed'].sum().sort_values(ascending = False).head(20).reset_index()

top_con_dth = top_date.groupby(by = 'Deaths')['Confirmed'].sum().sort_values(ascending = False).head(20).reset_index()

top_con_dth
top_cntry_con["Total Recovered"] = top_con_rec['Recovered']

top_cntry_con["Total Deaths"] = top_con_dth['Deaths']

top_cntry_con["Total Active Cases"]= top_con_act['Active']

top_cntry_con["Total Closed Cases"]= top_cntry_con['Total Confirmed']- top_con_act['Active']

top_cntry_con