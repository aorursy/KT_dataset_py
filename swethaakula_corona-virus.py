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
# importing the libraries

import pandas as pd

import matplotlib.pyplot as plt

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')
# importing the data

data=pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

data.head()
data.info()
#Changing the datatype of Date and Last Update columns

data['Date']=pd.to_datetime(data['Date'])

data['Last Update']=pd.to_datetime(data['Last Update'])

data.drop('Sno',axis=1,inplace=True)

data.head()
#Report on last update

from datetime import date

data_2_feb = data[data['Date'] > pd.Timestamp(date(2020,2,2))]

data_2_feb.head()
#Total countries effected till now



countries=data['Country'].unique().tolist()

print(countries)

print('\nTotal number of countries effected by corona till now: ',len(countries))
plt.bar(data['Date'],data['Confirmed'])

plt.xticks(rotation='vertical')

plt.xlabel('Dates')

plt.ylabel('confirmed cases')

plt.title('confirmed cases from January')

plt.show()
#According to the country how many are confirmed,death and Recovery cases



country_grp=data.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()

country_grp.head()
plt.plot(data['Date'],data['Deaths'])

plt.xticks(rotation='vertical')

plt.xlabel('Dates')

plt.ylabel('Death')

plt.title('Death cases from January')

plt.legend(loc='best',shadow='True')

plt.grid()

plt.show()


data.describe().T
#Recent cases

df_lastcases = data.groupby(['Country', 'Last Update']).sum().reset_index().sort_values('Last Update', ascending=False)

df_lastcases = df_lastcases.drop_duplicates(subset = ['Country'])

df_lastcases = df_lastcases[df_lastcases["Confirmed"]>0]

df_lastcases
#Confirmed cases around the world



fig = go.Figure()



fig.add_trace(go.Bar(x=df_lastcases["Confirmed"],

                y=df_lastcases['Country'],

                marker_color='green',

                marker_line_color='rgb(231,99,255)',

                marker_line_width=1.5,

                opacity=0.6,

                orientation='h')

             )



fig.update_layout(

    title_text='Last Confirmed Cases around the world ',

    height=700, width=800,

    showlegend=False

)



fig.show()
fig = go.Figure(data=[

    go.Bar(name='Recovered', x=df_lastcases['Country'], y=df_lastcases['Recovered'], marker_color='green'),

    go.Bar(name='Deaths', x=df_lastcases['Country'], y=df_lastcases['Deaths'], marker_color='red')

])



fig.update_traces(marker_line_color='rgb(48,88,107)',

                  marker_line_width=2, opacity=0.7

                 )



fig.update_layout(height=600, width=800, 

                  title="Death/Recovered Rate in the Other China Provinces"

                 )



fig.show()
fig = px.scatter_geo(df_lastcases, locations="Country", locationmode='country names', 

                     color="Confirmed", hover_name="Country" ,range_color= [0, 10], projection="natural earth",

                    title='Spread across the world')

fig.update(layout_coloraxis_showscale=False)

fig.show()
px.treemap(df_lastcases.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

           path=["Country"], values="Confirmed")