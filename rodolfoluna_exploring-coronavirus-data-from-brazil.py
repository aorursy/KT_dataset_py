import pandas as pd

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
df
df_ranked = df.groupby("state").last().drop(columns=['date']).sort_values(by=['cases'], ascending=False)

df_ranked
cases = df[['date','cases','deaths']].groupby('date').sum().reset_index()



#Starting the plotting after cases is more than 0.

cases = cases[(cases['cases'] > 0)].melt(id_vars =['date'], value_vars =['cases','deaths']) 



fig = px.line(cases, x='date', y='value', color='variable')

fig.update_layout(title='Data from Covid19 on Brazil overtime.',

                  xaxis_title='States', yaxis_title='Number of cases',legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.03,y=0.98))

fig.show()
states = df[(df['state'].isin([i for i in df_ranked.index[:5]]))]

states = states[(states['cases'] > 0)]



fig = px.line(states, x='date', y='cases', color='state')

fig.update_layout(title='Data from Covid19 on Brazil overtime(Cases).',

                  xaxis_title='Date', yaxis_title='Number of cases', legend_title='<b>Rank of top 5 states</b>',

                  legend=dict(x=0.03,y=0.98))

fig.show()
states = df[(df['state'].isin([i for i in df_ranked.index[:5]]))]

states = states[(states['deaths'] > 0)]



fig = px.line(states, x='date', y='deaths', color='state')

fig.update_layout(title='Data from Covid19 on Brazil overtime(Deaths).',

                  xaxis_title='Date', yaxis_title='Number of deaths', legend_title='<b>Rank of top 5 states</b>',

                  legend=dict(x=0.03,y=0.98))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Cases', x=df_ranked.index, y=df_ranked['cases']),

    go.Bar(name='Deaths', x=df_ranked.index, y=df_ranked['deaths'])

])

fig.update_layout(barmode='stack', title="COVID-19 in Brazil: number of cases by state", 

                  xaxis_title="States", yaxis_title="Number of cases", legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.90,y=0.5))

fig.show()
#Creating dataframe ranked by region.

df_ranked_region = df_ranked.groupby("region").agg({'cases':'sum', 'deaths':'sum'}).sort_values(by=['cases'], ascending=False)

df_ranked_region
plt.figure(figsize=(12,12))



df_ranked_region['cases'].plot( kind='pie'

                       , autopct='%1.1f%%'

                       , shadow=True

                       , startangle=10)



plt.title('Covid-19 Distribution - Cases on Brazil regions',size=25)

plt.legend(loc = "upper right"

           , fontsize = 10

           , ncol = 1 

           , fancybox = True

           , framealpha = 0.80

           , shadow = True

           , borderpad = 1);
plt.figure(figsize=(12,12))



df_ranked_region['deaths'].plot( kind='pie'

                       , autopct='%1.1f%%'

                       , shadow=True

                       , startangle=10)



plt.title('Covid-19 Distribution - Deaths toll on Brazil regions',size=25)

plt.legend(loc = "upper right"

           , fontsize = 10

           , ncol = 1 

           , fancybox = True

           , framealpha = 0.80

           , shadow = True

           , borderpad = 1);