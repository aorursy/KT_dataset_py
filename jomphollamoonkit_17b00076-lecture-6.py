import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
# print(df.columns)
# print(np.unique(df['Country/Region'].values))
country = 'UK'
# print(df[df['Country/Region']== country])
df1 = df[df['Country/Region']== country]
# print(df1)
print(df1.groupby('ObservationDate').sum())

df2 = df1.groupby('ObservationDate').sum()




# print(df2['Confirmed'].diff())
# diff = subtraction for daily basis
# df2 = cumulative data
df2['daily_confirmed'] = df2['Confirmed'].diff()
df2['daily_deaths'] = df2['Deaths'].diff()
df2['daily_recovered'] = df2['Recovered'].diff()
# print(df2)
df2['daily_confirmed'].plot()
df2['daily_recovered'].plot()
df2['daily_deaths'].plot()
plt.ylabel("Number of People")
plt.xlabel("Date")
plt.show()

#Cumulative plot
df2['Confirmed'].plot()
df2['Recovered'].plot()
df2['Deaths'].plot()
plt.ylabel("Number of People")
plt.xlabel("Date")
plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df2.index,y=df2['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df2.index,y=df2['daily_deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df2.index,y=df2['daily_recovered'].values,name='Daily recovered')

layout_object = go.Layout(title='UK daily cases 17B00076',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('UK_DAILY_CASES_17B00076.html')


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df2.index,y=df2['Confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df2.index,y=df2['Deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df2.index,y=df2['Recovered'].values,name='Daily recovered')

layout_object = go.Layout(title='UK daily cases 17B00076',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('UK_DAILY_CASES_17B00076.html')
# print(df2)
# df 3 = 
# df3 = df2[['daily_confirmed']]
df3 = df2
df3 = df3.fillna(0.)
# print(df3)
styled_object = df3.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('Table_17B00076.html','w')
f.write(styled_object.render())

# print df
date = '06/10/2020'
rank = df[df.index==date]
# df.index = row labels
rank = rank.groupby('Country/Region').sum()
rank = rank.sort_values(by='Confirmed',ascending=False).reset_index() 
print(rank)

print('Global rank of',country,'on',date,' = ', rank[rank['Country/Region']=='UK'].index.values [0]+1)


#read https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
