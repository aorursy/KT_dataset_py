# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# visualization

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns

from plotnine import *

import plotly.express as px

import folium





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





# color pallette

cdr = ['#393e46', '#ff2e63', '#30e3ca'] # grey - red - blue

idr = ['#f8b400', '#ff2e63', '#30e3ca'] # yellow - red - blue
COVID19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

COVID19.head()
COVID19.info()
# checking for missing value

COVID19.isna().sum()
# replacing Mainland china with just China

COVID19['Country'] = COVID19['Country'].replace('Mainland China', 'China')



# filling missing values with NA

COVID19[['Province/State']] = COVID19[['Province/State']].fillna('NA')
# Countries affected



countries = COVID19['Country'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
Number_of_countries = len(COVID19['Country'].value_counts())





situation = pd.DataFrame(COVID19.groupby('Country')['Confirmed'].sum())

situation['Country'] = situation.index

situation.index=np.arange(1, Number_of_countries + 1)



global_cases = situation[['Country','Confirmed']]

global_cases.sort_values(by=['Confirmed'],ascending=False)

provinces_situation = COVID19.groupby(['Country', 'Province/State'])['Confirmed', 'Deaths', 'Recovered'].max()

provinces_situation.style.background_gradient(cmap='viridis')



fig = px.bar(COVID19[['Country', 'Confirmed']].sort_values('Confirmed', ascending=False), 

             y="Confirmed", x="Country", color='Country', 

             log_y=True, template='ggplot2', title='Confirmed Cases')

fig.show()



fig = px.bar(COVID19[['Country', 'Deaths']].sort_values('Deaths', ascending=False), 

             y="Deaths", x="Country", color='Country', title='Deaths',

             log_y=True, template='ggplot2')

fig.show()
fig = px.choropleth(COVID19, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country", range_color=[1,2000], 

                    color_continuous_scale=px.colors.diverging.Tealrose, 

                    title='Countries with Confirmed Cases')

fig.update(layout_coloraxis_showscale=False)

fig.show()



# ------------------------------------------------------------------------



fig = px.choropleth(COVID19[COVID19['Deaths']>0], 

                    locations="Country", locationmode='country names',

                    color="Deaths", hover_name="Country", 

                    range_color=[1,50], color_continuous_scale=px.colors.sequential.Viridis,

                    title='Countries with Deaths Reported')

fig.update(layout_coloraxis_showscale=False)

fig.show()


