import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sbn



df = pd.read_csv('../input/parking-citations.csv')
#Validation dataframe

df_validation = df[["Issue Date","Location","Violation code","Fine amount"]].copy()

df_validation.dropna(axis='rows', inplace=True)

df_validation.shape
## Validating

#All dates are in format like 2015-12-21T00:00:00 using regex

assert False not in df_validation['Issue Date'].str.match(r"^[\d]{4}-[\d]{2}-[\d]{2}T.*$").values

print('Issue Date is OK')



#Location is string

assert pd.api.types.is_string_dtype(df_validation['Location'])

print('Location is OK')



#Violation code is string

assert pd.api.types.is_string_dtype(df_validation['Violation code'])

print('Violation code is OK')



#Fine amount is float

assert pd.api.types.is_float_dtype(df_validation['Fine amount'])

print('Fine amount is OK')
#Issue Date to datetime

df['Issue Date'] = df['Issue Date'].apply(lambda x: str(x).split('T')[0])

df['Issue Date'] = pd.to_datetime(df['Issue Date'], infer_datetime_format=True)

df.set_index(df["Issue Date"],inplace=True)

df.head()
import datetime as DT

today = DT.date.today()

month_ago = today - DT.timedelta(days=30)



#Monthly Number of Incidents

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

sbn.lineplot(data=df['Ticket number'].resample('D').count().truncate(before=month_ago), ax=ax)

ax.set(title='Daily Number of Incidents', xlabel='Time', ylabel='NO. of Incidents')

plt.show()

plt.rcParams['figure.figsize'] = 15,5
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



data = [go.Scatter(x=df['Ticket number'].resample('D').count().truncate(before=month_ago).index, y=df['Ticket number'].resample('D').count().truncate(before=month_ago))]



# specify the layout of our figure

layout = dict(title = "Daily Number of Incidents",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
#group by Location

df_group = (df.groupby('Location', as_index=True)).agg({'Ticket number':'count'}).rename(columns={'Ticket number': 'Incidents Size'})

#select top 10 location based on incidents

df_group = df_group.sort_values(by='Incidents Size', ascending=False).iloc[0:10, :]

df_group.head(10)
#Number of Incidents of top 10 locations

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

sbn.barplot(x=df_group.index, y=df_group['Incidents Size'], ax=ax)

ax.set(title='Incidents by Location', xlabel='Location', ylabel='NO. of Incidents')

plt.show()

plt.rcParams['figure.figsize'] = 28,5
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



data = [go.Bar(x=df_group.index, y=df_group['Incidents Size'])]



# specify the layout of our figure

layout = dict(title = "Incidents by Location",

              xaxis= dict(ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
#group by Violation code

df_group = (df.groupby('Violation code', as_index=True)).agg({'Ticket number':'count'}).rename(columns={'Ticket number': 'Incidents Size'})

#select top 10 location based on incidents

df_group = df_group.sort_values(by='Incidents Size', ascending=False).iloc[0:10, :]

df_group.head(10)
#Top 10 violation codes

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

sbn.barplot(x=df_group.index, y=df_group['Incidents Size'], ax=ax)

ax.set(title='Incidents by Violation Code', xlabel='Violation Code', ylabel='NO. of Incidents')

plt.show()

plt.rcParams['figure.figsize'] = 23,8
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



data = [go.Bar(x=df_group.index, y=df_group['Incidents Size'])]



# specify the layout of our figure

layout = dict(title = "Incidents by Violation Code",

              xaxis= dict(ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
#Monthly Amount Collected

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

sbn.lineplot(data=df['Fine amount'].resample('M').sum().truncate(before='2018'), ax=ax)

ax.set(title='Monthly Fine Amount', xlabel='Time', ylabel='Fine Amount')

plt.show()

plt.rcParams['figure.figsize'] = 15,5
# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



data = [go.Scatter(x=df['Fine amount'].resample('M').sum().truncate(before='2018').index, y=df['Fine amount'].resample('M').sum().truncate(before='2018'))]



# specify the layout of our figure

layout = dict(title = "Monthly Fine Amount",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)