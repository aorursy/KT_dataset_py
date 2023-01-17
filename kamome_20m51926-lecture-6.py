import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='South Korea'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

# df.index=pd.to_datetime(df.index).strftime('%m-%d')

df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()



# print(df)
df['daily_confirmed'].plot()

df['daily_recovery'].plot()

df['daily_deaths'].plot()

# plt.show()

plt.close()
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recoveries_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recoveries')



layout_object = go.Layout(title='Taiwan daily cases 20M51926',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)

iplot(fig)

fig.write_html('TW_daily_cases_20M51926.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

color="Oranges"#'gist_ncar'

styled_object = df1.style.background_gradient(cmap=color).highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51926.html','w')

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

latest_date=df["ObservationDate"].iloc[-1]

df1 = df[df["ObservationDate"]==latest_date].reset_index().iloc[:,1:]

df2 = df1.groupby(['Country/Region']).sum()

df3 = df2.sort_values(by=['Confirmed'],ascending=False).reset_index()



TW_r=df3.index[df3.iloc[:,0]=="Taiwan"].tolist()[0]+1

All_r=df3.index[-1]+1

print("Taiwan ranks %d / %d"%(TW_r,All_r))