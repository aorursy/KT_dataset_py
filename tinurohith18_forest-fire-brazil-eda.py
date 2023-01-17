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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

from pandas import DataFrame, Series

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

data=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', sep=',', encoding='latin')
# Total observations:

data.shape
#  Top 5 reasdings:

data.head()
# Check data-type:

data.dtypes
# Missing Values:

data.isnull().sum()
# Conversoin to datetime format:

data['date']=pd.to_datetime(data['date'])
# Unique state:

data['state'].unique()
# Start date:

data['date'].min()
# End date:

data['date'].min()
# Total count of forest fire occurance across each year:

a=data.groupby('year')['number'].sum()

a=pd.DataFrame(a)

a=a.sort_values(by='number', ascending=False)

plt.figure(figsize=(12,5))

sns.barplot(a.index,a.number)

plt.title('Total count of forest fire occurance for each year')
# Top 10 states - with forest fire alerts:

a=data.groupby('state')['number'].sum()

a=pd.DataFrame(a)

a=a.sort_values(by='number',ascending=False)

plt.figure(figsize=(12,7))

sns.barplot(a[0:10].index,a[0:10].number)

plt.title('Top 10 states- contributing to forest fire')
# Occurance of forest fire across months:

a=data.groupby('month')['number'].sum()

a=pd.DataFrame(a)

a=a.sort_values(by='number',ascending=False)

plt.figure(figsize=(12,5))

sns.barplot(a.index,a.number)

plt.title(' Occurance of forest fire across months')
import plotly

import plotly.graph_objs as go
# Interactive plot -  Occurance of forest fire for each month across 1998-2017:

a=pd.DataFrame(data.groupby('month')['number'].mean())

fig = go.Figure([go.Bar(x=pd.DataFrame(data[data['year']==i].groupby('month')['number'].mean()).index,

                        y=pd.DataFrame(data[data['year']==i].groupby('month')['number'].mean()).number, name=str(i)) 

                 for i in data.year.unique()],

               go.Layout(yaxis={'title': 'Average occurance of forest fire'},

                        title= ' Occurance of forest fire for each month across 1998-2017 '))

fig.show()
# Top 10 states - with forest fire report:

dat=data.groupby('state')['number'].sum().sort_values(ascending=False)

dat=pd.DataFrame(dat)

dat=dat[0:10]

dat.head(10)
# Occurance of forest fire for top 10 states

fig=go.Figure([go.Bar(x=dat.index,y=dat.number,marker_color=['green','yellow','blue','orange','purple','pink','violet','grey','red','magenta'])],

             go.Layout(xaxis={'title':'State'},

                        yaxis={'title':'Total count of forest fires'},

                        title=' Occurance of forest fire for top 10 states '))

fig.show()
# Total occurance of forest fire between 1998-2017

fig=go.Figure([go.Line(x=pd.DataFrame(data.groupby('year')['number'].sum()).index,

                       y=pd.DataFrame(data.groupby('year')['number'].sum()).number)],

             go.Layout(xaxis={'title':'Year'},

                        yaxis={'title':'Total number of  forest fires'},

                        title=' Total occurance of forest fire between 1998-2017'))

fig.show()
lat=[-16.350000, -22.15847, -23.533773, -22.908333, -11.409874, -21.5089, -16.328547,

     -19.841644, -21.175, -3.416843]

long=[-56.666668, -43.29321, -46.625290, -43.196388, -41.280857, -43.3228, -48.953403,

     -43.986511, -43.01778, -65.856064]

dat['lat']=lat

dat['lng']=long
dat.reset_index(inplace=True)

dat.head()
lat=[]

lon=[]

for i in data['state']:

    if i=='Acre':

        lat.append(-9.0238)

        lon.append(-70.8120)

    elif i=='Alagoas':

        lat.append(-9.5713)

        lon.append(-36.7820)

    elif i=='Amapa':

        lat.append(-0.9020)

        lon.append(-52.0030)

    elif i=='Amazonas':

        lat.append(-3.4168)

        lon.append(-65.8561)

    elif i=='Bahia':

        lat.append(-12.5797)

        lon.append(-41.7007)

    elif i=='Ceara':

        lat.append(-5.4984)

        lon.append(-39.3206)

    elif i=='Distrito Federal':

        lat.append(-15.7998)

        lon.append(-47.8645)

    elif i=='Espirito Santo':

        lat.append(-19.1834)

        lon.append(-40.3089)

    elif i=='Goias':

        lat.append(-15.8270)

        lon.append(-49.8362)

    elif i=='Maranhao':

        lat.append(-4.9609)

        lon.append(-45.2744)

    elif i=='Mato Grosso':

        lat.append(-12.6819)

        lon.append(-56.9211)

    elif i=='Minas Gerais':

        lat.append(-18.5122)

        lon.append(-44.5550)

    elif i=='Pará':

        lat.append(-1.9981)

        lon.append(-54.9306)

    elif i=='Paraiba':

        lat.append(-7.2400)

        lon.append(-36.7820)

    elif i=='Pernambuco':

        lat.append(-8.8137)

        lon.append(-36.9541)

    elif i=='Piau':    

        lat.append(-5.0920)

        lon.append(-42.8038)

    elif i=='Rio':

        lat.append(-22.9068)

        lon.append(-43.1729)

    elif i=='Rondonia':

        lat.append(-11.5057)

        lon.append(-63.5806)

    elif i=='Roraima':

        lat.append(-2.7376)

        lon.append(-62.0751)

    elif i=='Santa Catarina':

        lat.append(-27.2423)

        lon.append(-50.2189)

    elif i=='Sao Paulo':

        lat.append(-23.5505)

        lon.append(-46.6333)

    elif i=='Sergipe':

        lat.append(-10.5741)

        lon.append(-37.3857)

    else:

        lat.append(-10.1753)

        lon.append(-48.2982)

data['lat']=lat

data['lon']=lon
fig = go.Figure([go.Scattergeo(

        lon = data[data['state']==i]['lon'],

        lat = data[data['state']==i]['lat'],

        text = data[data['state']==i]['number'],

        mode='markers',

        marker=dict(size=12),

        name=str(i)

        )for i in data.state.unique()])



fig.update_layout(

        title = ' Forest fire at States of Brazil ',

        geo_scope='south america',

    )

fig.show()
# Total number of forest fire across each month for top 10 states

dat=data[data['state'].isin(['Mato Grosso','Paraiba','Sao Paulo','Rio','Bahia','Piau','Goias','Minas Gerais','Tocantins','Amazonas','Ceara'])]

fig = go.Figure([go.Bar(x=pd.DataFrame(dat[dat['state']==i].groupby('month')['number'].sum()).index,

                        y=pd.DataFrame(dat[dat['state']==i].groupby('month')['number'].sum()).number, name=i) for i in dat.state.unique()],

               go.Layout(xaxis={'title':'Months'},

                        yaxis={'title':'Total number of forest fire'},

                        title=' Total number of forest fire across each month for top 10 states'))

fig.show()
#  Total number of forest fire across 1998-2017 for top 10 states

dat=data[data['state'].isin(['Mato Grosso','Paraiba','Sao Paulo','Rio','Bahia','Piau','Goias','Minas Gerais','Tocantins','Amazonas','Ceara'])]

fig = go.Figure([go.Bar(x=pd.DataFrame(dat[dat['state']==i].groupby('year')['number'].sum()).index,

                        y=pd.DataFrame(dat[dat['state']==i].groupby('year')['number'].sum()).number, name=i) for i in dat.state.unique()],

               go.Layout(xaxis={'title':'Year'},

                        yaxis={'title':'Total number of forest fire'},

                        title=' Total number of forest fire across 1998-2017 for top 10 states'))

fig.show()
# Percentage distribution of forest fire across each year

a=pd.DataFrame(dat.groupby('state')['number'].sum())

fig = go.Figure([go.Pie(labels=a.index, values=a.number, hole=0.3)],

                go.Layout(title=' % distribution of forest fire across each year '))

fig.show()
# Percentage distribution of average forest fire across each month

a=pd.DataFrame(data.groupby('month')['number'].mean())

fig = go.Figure([go.Pie(labels=a.index, values=a.number, hole=0.3)],

               go.Layout(title=' % distribution of average forest fire across each month'))

fig.show()