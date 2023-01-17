import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header= 0)

df = df[df['Country/Region']=='Italy']
df = df.groupby('ObservationDate').sum()

df['Daily_Confirmed'] = df['Confirmed'].diff()
df['Daily_Deaths'] = df['Deaths'].diff()
df['Daily_Recovered'] = df['Recovered'].diff()
#print(df)
from plotly.offline import iplot
import plotly.graph_objs as go

day_conf = go.Scatter(x = df.index, y = df['Daily_Confirmed'].values, name = 'Daily Confirmed')
day_dths = go.Scatter(x = df.index, y = df['Daily_Deaths'].values, name = 'Daily Deaths')
day_reco = go.Scatter(x = df.index, y = df['Daily_Recovered'].values, name = 'Daily Recovered')

layout_obj = go.Layout(title = 'Daily Cases 19M58506 Italy', xaxis = dict(title='Date'), yaxis = dict(title = 'Number of people'))
fig = go.Figure(data = [day_conf, day_dths, day_reco], layout = layout_obj)

iplot(fig)
fig.write_html('Italy_daily_cases_19M58506.html')
df_c = df#[['Daily_Confirmed']]
df_c = df_c.fillna(0.)

styled_obj = df_c.style.background_gradient(cmap = 'rainbow').highlight_max('Daily_Confirmed').set_caption('Daily Summires')
display(styled_obj)
f = open('table_19M58506.html','w')
f.write(styled_obj.render())
df1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header= 0)

df2 = df1[df1['ObservationDate'] == '06/12/2020']
df1 = df2.groupby(['Country/Region']).sum()
df1 = df1.sort_values(by=['Confirmed'], ascending = False).reset_index()

df1['Rank'] = np.arange(1,len(df1)+1,1)
print(df1)

print(df1[df1['Country/Region'].isin(['Italy'])])
