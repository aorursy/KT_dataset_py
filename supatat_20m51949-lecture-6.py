import numpy as np # linear algebra

import matplotlib.pyplot as plt  # visualization

import pandas as pd  # data processing

np.set_printoptions(threshold=np.inf)



selected_country='Denmark'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0) #header=0 to set the first row as the header

#print(np.unique(df['Country/Region']cor.values))

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed']=df['Confirmed'].diff() #diff() is to substract the row before

df['daily_deaths']=df['Deaths'].diff()

df['daily_recovered']=df['Recovered'].diff()

df['daily_confirmed'].plot()

df['daily_deaths'].plot()

df['daily_recovered'].plot()

plt.show()
print(df)

from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovered'].values,name='Daily recovered')



layout_object = go.Layout(title=selected_country + ' daily cases 20M51949',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)

iplot(fig)

fig.write_html(selected_country + ' daily_cases_20M51949.html')
from plotly.offline import iplot

import plotly.graph_objs as go



confirmed_object = go.Scatter(x=df.index,y=df['Confirmed'].values,name='Cumulative confirmed')

deaths_object = go.Scatter(x=df.index,y=df['Deaths'].values,name='Cumulative deaths')

recovered_object = go.Scatter(x=df.index,y=df['Recovered'].values,name='Cumulative recovered')



layout_object = go.Layout(title=selected_country + ' cumulative cases 20M51949',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[confirmed_object,deaths_object,recovered_object],layout=layout_object)

iplot(fig)

fig.write_html(selected_country + ' daily_cases_20M51949.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51949.html','w')

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

latest = df['ObservationDate'].max() #To determine the latest date in dataframe

df = df[df['ObservationDate']==latest]

df = df.groupby(['Country/Region']).sum()

df = df.sort_values(by='Confirmed',ascending=False).reset_index() 

print(df)
print('Global ranking of '+selected_country+' is ', df[df['Country/Region']==selected_country].index.values[0]+1) #To extract the index value of row with the selected country, and plus 1 (since index starts from 0)
ranking = 100 #To show the country at the specific ranking

df.loc[ranking]