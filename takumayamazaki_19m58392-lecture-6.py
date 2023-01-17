import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


selected_country='Netherlands'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(df['Country/Region'].values)
df=df[df['Country/Region']==selected_country]
df=df.groupby('ObservationDate').sum()
print(df)
df['daily_confirmed']= df['Confirmed'].diff()
df['daily_deaths']= df['Deaths'].diff()
df['daily_recovery']= df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovery'].plot()
plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object=go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object=go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object=go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovery')

layout_object=go.Layout(title='Netherlands daily cases 19M58392',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig=go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Netherlands_cases_19M58392.html')
df1=df#[['daily_confirmed']]
df1=df1.fillna(0.)
styled_object=df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f=open('table_19M58392.html','w')
f.write(styled_object.render())
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1=df.groupby(['ObservationDate','Country/Region']).sum()
df2=df[df['ObservationDate']=='06/06/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
df=df2.groupby(['Country/Region']).sum()
print('Ranking of Netherlands:',df2[df2['Country/Region']=='Netherlands'].index.values[0]+1)
