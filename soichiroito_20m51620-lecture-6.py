import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

#Focus on Japan
selected_country='Japan'
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df=df[df['Country/Region']==selected_country]
df=df.groupby('ObservationDate').sum()
print(df)

#Calculate daily changes
df['daily_confirmed']=(df['Confirmed'].diff())
df['daily_deaths']=(df['Deaths'].diff())
df['daily_recovered']=(df['Recovered'].diff())

#Simple plot
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovered'].plot()
plt.show()
#To check the new column
print(df)
#Create dynamic figure
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovered'].values,name='Daily recovered')

layout_object=go.Layout(title='Japan daily cases 20M51620',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig=go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('Japan_daily_cases_20M51620.html')
#Create colored Table
df1=df

#To vanish NAN data
df1=df1.fillna(0.)

styled_object=df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily summaries')
display(styled_object)

f=open('table_20M51620.html','w')
f.write(styled_object.render())
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#Ranking of cumulative infections as of June 12, 2020
df=df[df['ObservationDate']=='06/12/2020']
df=df.groupby('Country/Region').sum()
df = df.sort_values(by='Confirmed',ascending=False).reset_index()

print('Ranking of Japan: ', df[df['Country/Region']=='Japan'].index.values[0]+1)

df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#Ranking of dead people as of June 12, 2020
df=df[df['ObservationDate']=='06/12/2020']
df=df.groupby('Country/Region').sum()
df = df.sort_values(by='Deaths',ascending=False).reset_index()
#print(df)

print('Ranking of Japan: ', df[df['Country/Region']=='Japan'].index.values[0]+1)
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#Ranking of recovered people as of June 12, 2020
df=df[df['ObservationDate']=='06/12/2020']
df=df.groupby('Country/Region').sum()
df = df.sort_values(by='Recovered',ascending=False).reset_index()
#print(df)

print('Ranking of Japan: ', df[df['Country/Region']=='Japan'].index.values[0]+1)
#More analysis in Japan
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

#Ranking of cumulative infections as of June 12, 2020 in Japan
df=df[df['ObservationDate']=='06/12/2020']
df=df[df['Country/Region']=='Japan']
df = df.sort_values(by='Confirmed',ascending=False).reset_index()
#print(df)

#Make pie chart
data=np.array(df['Confirmed'])
label=np.array(df['Province/State'])
plt.pie(data,labels=label)
plt.show()
