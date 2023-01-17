# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading Datasets From CSV Files

covid19_India = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")



# Show Random Records

covid19_India.sample(10)
# Getting Column Names

print("Column Names are : \n", covid19_India.columns)
#Dropping useless columns like "Sno"

covid19_India.drop(['Sno'], axis = 1, inplace = True)

covid19_India.sample(10)
#Analyse Date Wise

covid19_India['Confirmed'] = covid19_India['ConfirmedIndianNational']+covid19_India['ConfirmedForeignNational']

date_data = covid19_India[["Date","Confirmed","Deaths","Cured"]]

date_data['Date'] = date_data['Date'].apply(pd.to_datetime, dayfirst=True)

date_data
date_data = date_data.groupby(["Date"]).sum().reset_index() #This helps to get the remaining non editted indexes

date_data
from datetime import date

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Markdown

import plotly.graph_objs as go

import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.express as px

from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs

import plotly as ply

import pycountry

import folium 

from folium import plugins

import json

# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow
plot = date_data.melt(id_vars="Date", value_vars=['Cured', 'Deaths', 'Confirmed'],

                 var_name='Case', value_name='Count')



fig = px.area(plot, x="Date", y="Count", color='Case',title='Time wise cases analysis', color_discrete_sequence = [rec, dth, cnf])

fig.show()
#Deaths vs Recovered in India

covid19_India["Confirmed"]= covid19_India["ConfirmedIndianNational"]+covid19_India["ConfirmedForeignNational"]

d_r = covid19_India[['State/UnionTerritory','Confirmed','Deaths','Cured']]

fig2 = px.scatter(d_r , y = "Deaths", x ="Cured",color = "State/UnionTerritory",size = 'Confirmed',hover_data=['State/UnionTerritory','Confirmed','Deaths','Cured'],log_x=True,log_y=True)

fig2.show()
#Using Gradients

covid19_India.style.background_gradient(cmap='Reds')
Cure_over_Death = date_data.groupby('Date').sum().reset_index()



Cure_over_Death['No. of Deaths to 100 Confirmed Cases'] = round(Cure_over_Death['Deaths']/(Cure_over_Death['Confirmed']),3)*100

Cure_over_Death['No. of Recovered to 100 Confirmed Cases'] = round(Cure_over_Death['Cured']/(Cure_over_Death['Confirmed']),3)*100

Cure_over_Death = Cure_over_Death.melt(id_vars ='Date',

                          value_vars=['No. of Deaths to 100 Confirmed Cases','No. of Recovered to 100 Confirmed Cases'],

                          var_name='Ratio',

                          value_name='Value')



fig3 = px.line(Cure_over_Death, x='Date', y='Value', color='Ratio', log_y=True,

             title='Cure VS Death', color_discrete_sequence=[rec, dth])

fig3.show()