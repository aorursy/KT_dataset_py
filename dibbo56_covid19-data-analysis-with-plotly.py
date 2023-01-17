import datetime

import plotly.offline as pyo

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/covid19/train_data.csv')

df.head()
print('Data information')

print(df.info(), end='\n\n')

print('Checking for null values')

print(df.isnull().sum(), end='\n\n')

print('Necessary information from the dataset')

print('Total affected countries ', len(df['Country_Region'].unique()))

print('Total confirmed cases ', df['ConfirmedCases'].sum())

print('Total fatalities cases ', df['Fatalities'].sum())
d = df['Date'].unique()

date = {}

for i in d:

    date.update({i:0})

    

for i in date:

    date.update({i:df[df['Date']==i]['ConfirmedCases'].sum()})
x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in date.keys()]

y_values = date.values()
x_values = [i for i in range(1,82)]

y_values = [i for i in y_values]



fig = px.line(x=x_values, y=y_values, title='Global Confirmed Cases', labels={'x':'Days', 'y':'Confirmed Case'},height=600,)

fig.show()
d = df['Date'].unique()

date = {}

for i in d:

    date.update({i:0})

    

for i in date:

    date.update({i:df[df['Date']==i]['Fatalities'].sum()})
x_values = [datetime.datetime.strptime(d,"%Y-%m-%d").date() for d in date.keys()]

y_values = date.values()
x_values = [i for i in range(1,82)]

y_values = [i for i in y_values]



fig = px.line(x=x_values, y=y_values, title='Global Fatality Cases', labels={'x':'Days', 'y':'Fatality Case'},height=600,)

fig.show()
data = {'Country':[], 'ConfirmedCases':[], 'Fatalities':[]}

data.update({'Country':df['Country_Region'].unique()})



confirm_case = []

for i in data['Country']:

    confirm_case.append(df[df['Country_Region'] == i]['ConfirmedCases'].sum())



fatalities_case = []

for i in data['Country']:

    fatalities_case.append(df[df['Country_Region'] == i]['Fatalities'].sum())

    

data.update({'ConfirmedCases':confirm_case})

data.update({'Fatalities':fatalities_case})
data = pd.DataFrame(data)

data.head()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=False)
df_confirm_asc = df_confirm_asc.reset_index(drop=True)

df_confirm_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]

y_values = y_values[::-1]

x_values = x_values[::-1]

df1 = {'Country':x_values, 'ConfirmedCases':y_values}

df1 = pd.DataFrame(df1)
fig = px.bar(df1, x='ConfirmedCases', y='Country',  color_discrete_sequence=["red"]*10, title='Top 10 Highest Confirmed Cases Country', barmode="group")

fig.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=False)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

df_fatality_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_fatality_asc.loc[0:9,'Country']]

y_values = [i for i in df_fatality_asc.loc[0:9,'Fatalities']]

x_values = x_values[::-1]

y_values = y_values[::-1]

df1 = {'Country':x_values, 'Fatality Cases':y_values}

df1 = pd.DataFrame(df1)
fig = px.bar(df1, x='Fatality Cases', y='Country',  color_discrete_sequence=["red"]*10, title='Top 10 Highest Fatality Cases Country', barmode="group")

fig.show()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=True)

df_confirm_asc = df_confirm_asc.reset_index(drop=True)

df_confirm_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]

x_values = x_values[::-1]

y_values = y_values[::-1]

df1 = {'Country':x_values, 'ConfirmedCases':y_values}

df1 = pd.DataFrame(df1)
fig = px.bar(df1, x='ConfirmedCases', y='Country',  color_discrete_sequence=["red"]*10, title='10 Lowest Confirmed Cases Country', barmode="group")

fig.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=True)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

df_fatality_asc.style.background_gradient(cmap="Reds")
x_values = [i for i in df_fatality_asc.loc[34:43,'Country']]

y_values = [i for i in df_fatality_asc.loc[34:43,'Fatalities']]

x_values = x_values[::-1]

y_values = y_values[::-1]

df1 = {'Country':x_values, 'Fatalities':y_values}

df1 = pd.DataFrame(df1)
fig = px.bar(df1, x='Fatalities', y='Country',  color_discrete_sequence=["red"]*10, title='10 Lowest Fatality Cases Country', barmode="group")

fig.show()
df_confirm_asc = data.sort_values(by=['ConfirmedCases'], ascending=False)

df_confirm_asc = df_confirm_asc.reset_index(drop=True)

x_values = [i for i in df_confirm_asc.loc[0:9,'Country']]

y_values = [i for i in df_confirm_asc.loc[0:9,'ConfirmedCases']]

df1 = {'Country':x_values, 'ConfirmedCases':y_values}

df1 = pd.DataFrame(df1)



fig = px.pie(df1, values='ConfirmedCases', names='Country', title='Top 10 Highest Confirmed Cases Country', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
df_fatality_asc = data.sort_values(by=['Fatalities'], ascending=False)

df_fatality_asc = df_fatality_asc.reset_index(drop=True)

x_values = [i for i in df_fatality_asc.loc[0:9,'Country']]

y_values = [i for i in df_fatality_asc.loc[0:9,'Fatalities']]

df1 = {'Country':x_values, 'Fatalities':y_values}

df1 = pd.DataFrame(df1)



fig = px.pie(df1, values='Fatalities', names='Country', title='Top 10 Highest Fatality Cases Country', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
data = {'Date':[], 'US':[], 'China':[], 'Italy':[], 'Spain':[], 'Germany':[], 'France':[], 'Iran':[]}

data.update({'Date': df['Date'].unique()})



for i in data['Date']:

    data['US'].append(df[(df['Date']==i) & (df['Country_Region']=='US')]['ConfirmedCases'].sum())

    data['China'].append(df[(df['Date']==i) & (df['Country_Region']=='China')]['ConfirmedCases'].sum())

    data['Italy'].append(df[(df['Date']==i) & (df['Country_Region']=='Italy')]['ConfirmedCases'].sum())

    data['Spain'].append(df[(df['Date']==i) & (df['Country_Region']=='Spain')]['ConfirmedCases'].sum())

    data['Germany'].append(df[(df['Date']==i) & (df['Country_Region']=='Germany')]['ConfirmedCases'].sum())

    data['France'].append(df[(df['Date']==i) & (df['Country_Region']=='France')]['ConfirmedCases'].sum())

    data['Iran'].append(df[(df['Date']==i) & (df['Country_Region']=='Iran')]['ConfirmedCases'].sum())

    

data = pd.DataFrame(data)

data.head() 
df_long=pd.melt(data, id_vars=['Date'], value_vars=['US', 'China', 'Italy', 'Spain', 'Germany','France','Iran'])

fig = px.line(df_long, x='Date', y='value', color='variable', labels={'Date':'Date', 'value':'Confirmed Case'} ,title = 'US, Chian, Italy, Spain, Germany, France, Iran Confirmed Cases \nFrom 2020/01/22 to 2020/04/11')

fig.show()
data = {'Date':[], 'US':[], 'China':[], 'Italy':[], 'Spain':[], 'Germany':[], 'France':[], 'Iran':[]}

data.update({'Date': df['Date'].unique()})



for i in data['Date']:

    data['US'].append(df[(df['Date']==i) & (df['Country_Region']=='US')]['Fatalities'].sum())

    data['China'].append(df[(df['Date']==i) & (df['Country_Region']=='China')]['Fatalities'].sum())

    data['Italy'].append(df[(df['Date']==i) & (df['Country_Region']=='Italy')]['Fatalities'].sum())

    data['Spain'].append(df[(df['Date']==i) & (df['Country_Region']=='Spain')]['Fatalities'].sum())

    data['Germany'].append(df[(df['Date']==i) & (df['Country_Region']=='Germany')]['Fatalities'].sum())

    data['France'].append(df[(df['Date']==i) & (df['Country_Region']=='France')]['Fatalities'].sum())

    data['Iran'].append(df[(df['Date']==i) & (df['Country_Region']=='Iran')]['Fatalities'].sum())

    

data = pd.DataFrame(data)

data.head()
df_long=pd.melt(data, id_vars=['Date'], value_vars=['US', 'China', 'Italy', 'Spain', 'Germany','France','Iran'])

fig = px.line(df_long, x='Date', y='value', color='variable', labels={'Date':'Date', 'value':'Fatality Case'} ,title = 'US, Chian, Italy, Spain, Germany, France, Iran Confirmed Cases \nFrom 2020/01/22 to 2020/04/11')

fig.show()