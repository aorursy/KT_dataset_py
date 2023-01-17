import pandas as pd # Load data

import numpy as np # Scientific Computing

import seaborn as sns

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode,plot, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings # Ignore Warnings

warnings.filterwarnings("ignore")

sns.set() # Set Graphs Background
data = pd.read_csv('../input/datafileee/train (3).csv')

data.head()
data.info()
data['Country_Region'].unique()
data['Country_Region'].nunique()
data['Country_Region'].value_counts()
print(data['Date'].min())

print(data['Date'].max())
data_15 = data.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()

data_15 = data_15.nlargest(15,'ConfirmedCases')

data_15
data_15['Active/Recover'] = data_15['ConfirmedCases'] - data_15['Fatalities']

data_15
p_15 = data_15['Country_Region']

q_15 = data_15['ConfirmedCases']



data1 = go.Bar(x=p_15,y=q_15,name='ConfirmedCases')



layout = go.Layout(title='Country_Region VS ConfirmedCases',xaxis_title='Country_Region',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['Fatalities']



data1 = go.Bar(x=p_15,y=q_15)



layout = go.Layout(title='Country_Region VS Deaths',xaxis_title='Country_Region',yaxis_title='Deaths')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['Active/Recover']



data1 = go.Bar(x=p_15,y=q_15)



layout = go.Layout(title='Country_Region VS Active/Recover',

                   xaxis_title='Country_Region',yaxis_title='Active/Recover')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['Active/Recover']

z_15 = data_15['Fatalities']

m_15 = data_15['ConfirmedCases']



data1 = go.Bar(x=p_15,y=m_15,name='ConfirmedCases')

data2 = go.Bar(x=p_15,y=q_15,name='Active/Recover')

data3 = go.Bar(x=p_15,y=z_15,name='Deaths')



layout = go.Layout(barmode='group',title='ConfirmedCases & Active/Recover & Deaths VS Country_Region',

                   xaxis_title='Country_Region',

                   yaxis_title='ConfirmedCases & Active/Recover & Deaths')

data=[data1,data2,data3]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['ConfirmedCases']



data1 = go.Scatter(x=p_15,y=q_15,mode='markers')



layout = go.Layout(title='Country_Region VS ConfirmedCases',

                   xaxis_title='Country_Region',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['Active/Recover']



data1 = go.Scatter(x=p_15,y=q_15,mode='markers')



layout = go.Layout(title='Country_Region VS Active/Recover',

                   xaxis_title='Country_Region',yaxis_title='Active/Recover')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_15 = data_15['Country_Region']

q_15 = data_15['Fatalities']



data1 = go.Scatter(x=p_15,y=q_15,mode='markers')



layout = go.Layout(title='Country_Region VS Deaths',

                   xaxis_title='Country_Region',yaxis_title='Deaths')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
pp = data_15['Country_Region']

qq = data_15['ConfirmedCases']



data = go.Pie(labels=pp,values=qq, 

              hoverinfo='label+percent', textinfo='value')



layout = go.Layout(title='Country_Region VS ConfirmedCases')

data =[data]

fig = go.Figure(data,layout)

iplot(fig)
pp = data_15['Country_Region']

qq = data_15['Fatalities']



data = go.Pie(labels=pp,values=qq, 

              hoverinfo='label+percent', textinfo='value')



layout = go.Layout(title='Country_Region VS Fatalities')

data =[data]

fig = go.Figure(data,layout)

iplot(fig)
data1 = pd.read_csv('../input/datafileee/train (3).csv')

data1['Date'] = pd.to_datetime(data1['Date'])
data_81 = data1.groupby('Date', as_index=False)['ConfirmedCases','Fatalities'].sum()

data_81 = data_81.nlargest(81,'ConfirmedCases')

data_81
data_81['Active/Recover'] = data_81['ConfirmedCases'] - data_81['Fatalities']

data_81
p_81 = data_81['Date']

q_81 = data_81['ConfirmedCases']



data1 = go.Scatter(x=p_81,y=q_81)



layout = go.Layout(title='Date VS ConfirmedCases',

                   xaxis_title='Date',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['Active/Recover']



data1 = go.Scatter(x=p_81,y=q_81)



layout = go.Layout(title='Date VS Active/Recover',

                   xaxis_title='Date',yaxis_title='Active/Recover')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['Fatalities']



data1 = go.Scatter(x=p_81,y=q_81,mode='markers')



layout = go.Layout(title='Date VS Fatalities',

                   xaxis_title='Date',yaxis_title='Fatalities')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['ConfirmedCases']

z_81 = data_81['Active/Recover']

m_81 = data_81['Fatalities']



trace0 = go.Scatter(x=p_81,y=q_81,mode='markers')

trace1 = go.Scatter(x=p_81,y=z_81,mode='markers')

trace2 = go.Scatter(x=p_81,y=m_81,mode='markers')



layout = go.Layout(title='Total ConfirmedCases & Active/Recover & Deaths By Date',

                   xaxis_title='Date',yaxis_title='ConfirmedCases Active/Recover & Deaths')

data=[trace0,trace1,trace2]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['ConfirmedCases']



data1 = go.Bar(x=p_81,y=q_81)



layout = go.Layout(title='Date VS ConfirmedCases',

                   xaxis_title='Date',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['Fatalities']



data1 = go.Bar(x=p_81,y=q_81)



layout = go.Layout(title='Date VS Deaths',

                   xaxis_title='Date',yaxis_title='Deaths')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_81 = data_81['Date']

q_81 = data_81['ConfirmedCases']



data1 = go.Scatter(x=p_81,y=q_81)



layout = go.Layout(title='Date VS ConfirmedCases',

                   xaxis_title='Date',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
data2 = pd.read_csv('../input/datafileee/train (3).csv')

data_all = data2.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()

data_all
data_all['Active/Recover'] = data_all['ConfirmedCases'] - data_all['Fatalities']

data_all
data_usa = data_all.query("Country_Region=='US'")

data_usa
p_usa = data_usa['Date']

q_usa = data_usa['ConfirmedCases']



data1 = go.Scatter(x=p_usa,y=q_usa,mode='markers')



layout = go.Layout(title='Total ConfirmedCases By Date For United State',

                   xaxis_title='Date',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_usa = data_usa['Date']

q_usa = data_usa['Fatalities']



data1 = go.Scatter(x=p_usa,y=q_usa,mode='markers')



layout = go.Layout(title='Total Fatalities By Date For United State',

                   xaxis_title='Date',yaxis_title='Fatalities')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_usa = data_usa['Date']

q_usa = data_usa['ConfirmedCases']

z_usa = data_usa['Active/Recover']

m_usa = data_usa['Fatalities']



trace0 = go.Scatter(x=p_usa,y=q_usa,mode='markers')

trace1 = go.Scatter(x=p_usa,y=z_usa,mode='markers')

trace2 = go.Scatter(x=p_usa,y=m_usa,mode='markers')



layout = go.Layout(title='Total ConfirmedCases & Active/Recover & Deaths By Date',

                   xaxis_title='Date',yaxis_title='ConfirmedCases Active/Recover & Deaths')

data=[trace0,trace1,trace2]

fig = go.Figure(data,layout)



iplot(fig)
p_usa = data_usa['Date']

q_usa = data_usa['ConfirmedCases']



data1 = go.Bar(x=p_usa,y=q_usa)



layout = go.Layout(title='Date VS ConfirmedCases US',

                   xaxis_title='Date',yaxis_title='ConfirmedCases')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)
p_usa = data_usa['Date']

q_usa = data_usa['Fatalities']



data1 = go.Bar(x=p_usa,y=q_usa)



layout = go.Layout(title='Date VS Deaths US',

                   xaxis_title='Date',yaxis_title='Deaths')

data=[data1]

fig = go.Figure(data,layout)



iplot(fig)