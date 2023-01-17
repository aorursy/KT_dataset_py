# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

from datetime import datetime

import matplotlib.pylab as plt

from statsmodels.tsa.stattools import adfuller

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import acf, pacf







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# remove unnecessary columns

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df = df.drop(['SNo','Last Update'], axis=1)

df.ObservationDate = df.ObservationDate.apply(pd.to_datetime)

df.sort_values(by='ObservationDate',ascending=False)
df.ObservationDate.unique()  

df.ObservationDate.isnull().any() 



df[['Confirmed','Deaths','Recovered']].isnull().any() 

df['Country/Region'].isnull().any()  

df['Province/State'].isnull().any()



# Found no missing values in any of these colums
# Checking if any invalid / negative cases

df[df['Confirmed'] < 0 ]  

df[df['Deaths'] < 0 ] 

df[df['Recovered'] < 0 ]  



# rename countries and provinces

df["Country/Region"].replace({"Iran (Islamic Republic of)": "Iran", "Viet Nam":"Vietnam"}, inplace=True)



# list of countries without provinces/state

df[df['Province/State'].isnull()]['Country/Region'].unique()
# list of countries with provinces/state

# the 2 lists of countries with and without provinces/state are mutually exclusive. No incorrect entries or errors in country names



df['Country/Region'].replace({"Taipei and environs": "Taiwan"}, inplace=True)

df[~df['Province/State'].isnull()]['Country/Region'].unique() 
df.groupby(['Country/Region','Province/State']).size().head(50)
global_df = df.groupby('ObservationDate').sum()

global_df['death rate'] = round(global_df['Deaths'] / global_df['Confirmed'],4)*100



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=global_df.index, y=global_df['death rate'],mode='lines',name='Mortality Rate',line=dict(color='red', 

                                                                                                                           width=2.5)))

fig.update_layout(title="Global Death Rate",xaxis_title="Death Rate",yaxis_title="Date",font=dict(family="Arial",size=14,color='#F7E13F'))



fig.show()
global_df = df.groupby('ObservationDate').sum()



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=global_df.index, y=global_df['Deaths'],mode='lines',name='Deaths',line=dict(color='#D33D4D', width=3.5)))



fig.add_trace(go.Scatter(x=global_df.index, y=global_df['Confirmed'],mode='lines',name='Confirmed',line=dict(color='#3D83D3', 

                                                                                                                     width=1.5)))



fig.add_trace(go.Scatter(x=global_df.index, y=global_df['Recovered'],mode='lines',name='Recovered',line=dict(color='#77D33D', 

                                                                                                                     width=5)))





fig.update_layout(title="Global Number of Confirmed/Death/Recovered cases",xaxis_title="Number of cases",yaxis_title="Date",

                  font=dict(family="Robotic",size=14,color="white"))

fig.show()
country_unique_df = df.groupby('ObservationDate')['Country/Region'].nunique()

country_unique_df = pd.DataFrame({'ObservationDate':country_unique_df.index, 'Country Number':country_unique_df.values})

country_unique_df





fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=country_unique_df.ObservationDate, y=country_unique_df['Country Number'],mode='lines',

                         name='Number of unique countries infected with COVID-19',line=dict(color='#9D0369', width=5)))





fig.update_layout(title="Number of unique countries infected with COVID-19",yaxis_title="Number of Countries",xaxis_title="Date",

                  font=dict(family="Calibiri",size=16,color="white"))

fig.show()