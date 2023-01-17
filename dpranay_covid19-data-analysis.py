# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)



init_notebook_mode(connected=True)

cf.go_offline()



from plotly.subplots import make_subplots

import plotly.graph_objects as go



from IPython.core.display import HTML







import warnings

warnings.filterwarnings('ignore')
import requests





url = "https://www.worldometers.info/coronavirus/#countries"



header = {

  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",

  "X-Requested-With": "XMLHttpRequest"

}

r = requests.get(url, headers=header)



dfs = pd.read_html(r.text)

df = dfs[0]



time_series = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv',parse_dates=['Date'])



type(df)
df.isnull().sum()
# renaming the columns

df.rename(columns = {'Country,Other':'Country'},inplace = True)

df.rename(columns = {'Serious,Critical':'Serious'},inplace = True)

df.rename(columns = {'Tot Cases/1M pop':'TotalCases/1M'},inplace = True)

time_series.rename(columns = {'Country/Region':'Country'},inplace = True)

df.drop('#',axis =1,inplace = True)

df.head()

#drop the columns in time_series

time_series.drop('Province/State',axis =1,inplace = True)

time_series.head()
# Lets check the data types and change it

df.dtypes
#changing the data type



df['TotalCases'] = df['TotalCases'].fillna(0).astype('int')

df['TotalDeaths'] = df['TotalDeaths'].fillna(0).astype('int')

df['TotalRecovered'] = df['TotalRecovered'].fillna(0).astype('int')

df['ActiveCases'] = df['ActiveCases'].fillna(0).astype('int')

df['Serious'] = df['Serious'].fillna(0).astype('int')

df['Deaths/1M pop'] = df['Deaths/1M pop'].fillna(0).astype('int')

df['TotalTests'] = df['TotalTests'].fillna(0).astype('int')

df['Tests/ 1M pop'] = df['Tests/ 1M pop'].fillna(0).astype('int')

df['NewCases'] = df['NewCases'].fillna(0)

df[['NewCases']] = df[['NewCases']].replace('[\+,]', '', regex=True).astype(int)

df['NewDeaths'] = df['NewDeaths'].fillna(0)

df[['NewDeaths']] = df[['NewDeaths']].replace('[\+,]', '', regex=True).astype(int)

df[['Population']] = df[['Population']].fillna(0).astype('int')

time_series.fillna(0)

time_series.isnull().sum()

#highlighting the most no of cases

dataframe = df.iloc[1:216,:-1]





dataframe.style.background_gradient(cmap = 'Reds')
group1 = time_series.groupby(['Date', 'Country'])['Confirmed', 'Deaths','Recovered'].sum().reset_index()

heat= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Confirmed"]), 

                    hover_name="Country",projection = 'natural earth',title='Heatmap', color_continuous_scale=px.colors.sequential.Blues)



heat.update(layout_coloraxis_showscale=False)

heat.show()
group1
fig_heat= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Deaths"]), 

                    hover_name="Country",projection = 'natural earth',title='Heatmap(Deaths)', 

                    color_continuous_scale=px.colors.sequential.Reds)



fig_heat.update(layout_coloraxis_showscale=False)



fig_heat.show()
dataframe.head()


fig_z = px.bar(dataframe.sort_values('TotalCases'),x='TotalCases', y='Country',orientation = 'h',



            color_discrete_sequence=['#B3611A'],text = 'TotalCases',title='TotalCases')





fig_x = px.bar(dataframe.sort_values('TotalDeaths'),x='TotalDeaths', y='Country',orientation = 'h',

               color_discrete_sequence=['#830707'],text = 'TotalDeaths',title = 'TotalDeaths')





fig_ = px.bar(dataframe.sort_values('TotalRecovered'),x='TotalRecovered',y='Country',orientation ='h',

               color_discrete_sequence=['#073707'],text = 'TotalRecovered',title = 'TotalRecovered')



fig_p = make_subplots(rows =1,cols =3,subplot_titles=('TotalCases','TotalDeaths','TotalRecovered'))



fig_p.add_trace(fig_z['data'][0],row = 1,col =1)

fig_p.add_trace(fig_x['data'][0],row = 1,col =2)

fig_p.add_trace(fig_['data'][0],row=1,col=3)



fig_p.update_layout(height=3000,title ='Per Country')

fig_p.show()
#Top 20 countries with TotalCases

totalCases = df.iloc[1:21,0:2]



df1 = df[['Country','TotalDeaths']]

TotalDeaths = df1[1:21]



df2 = df[['Country','TotalRecovered']]

totalrecovered = df2[1:21]
data = totalCases.sort_values('TotalCases')

data1 = TotalDeaths.sort_values('TotalDeaths')

data2 = totalrecovered.sort_values('TotalRecovered')



fig1 = px.bar(data,x="TotalCases", y="Country",orientation = 'h',color_discrete_sequence=['#B3611A'],text='TotalCases')



fig2 = px.bar(data1,x="TotalDeaths", y="Country",orientation = 'h',color_discrete_sequence =['#830707'],text = 'TotalDeaths')



fig3 = px.bar(data2,x='TotalRecovered',y='Country',orientation = 'h',color_discrete_sequence=['#073707'],text = 'TotalRecovered')







fig = make_subplots(

    rows=2, cols=3,

    subplot_titles=("Totalconfirmed", "Total deaths", "total Recovered"))



fig.add_trace(fig1['data'][0], row=1, col=1)

fig.add_trace(fig2['data'][0], row=1, col=2)

fig.add_trace(fig3['data'][0], row=1, col=3)



fig.update_layout(height=1200,title = 'Top 20 Countries')







fig.show()
df3 = df[['Country','ActiveCases']]

ActiveCases = df3[1:21]



df4 = df[["Country","Serious"]]

Serious = df4[1:21]



df5 = df[['Country','TotalTests']]

TotalTests = df5[1:21]



df6 = df[['Country','Tests/ 1M pop']]

TestperM = df6[1:21]



df7 = df[['Country','NewDeaths']]

NewDeaths = df7[2:22]



df8 = df[['Country','Deaths/1M pop']]

DeathsperM = df8[1:21]



df9 = df[['Country','NewCases']]

NewCases = df9[1:22]











data3 = ActiveCases.sort_values('ActiveCases')

data4 = Serious.sort_values('Serious')





fig_q = px.bar(data3,x = 'ActiveCases',y='Country',orientation = 'h',color_discrete_sequence=['#476CC3'],

              text = 'ActiveCases')



fig_r = px.bar(data4, x = 'Serious', y='Country',orientation ='h',color_discrete_sequence=['#606060'],

              text = 'Serious')



fig_s = px.bar(TotalTests.sort_values('TotalTests'),x = 'TotalTests',y='Country',orientation = 'h',color_discrete_sequence=['#803604'],

              text = 'TotalTests')



fig_t = px.bar(TestperM.sort_values('Tests/ 1M pop'), x = 'Tests/ 1M pop', y='Country',orientation ='h',color_discrete_sequence=['#ff6666'],

              text = 'Tests/ 1M pop')



fig_u = px.bar(NewDeaths.sort_values('NewDeaths'), x = 'NewDeaths',y = 'Country',orientation = 'h',text = 'NewDeaths',

              color_discrete_sequence=['#660000'])



fig_i= px.bar(DeathsperM.sort_values('Deaths/1M pop'), x = 'Deaths/1M pop',y = 'Country', orientation ='h', text = 'Deaths/1M pop',

              color_discrete_sequence=['#663300'])







fig_o = px.bar(NewCases.sort_values('NewCases'),x= 'NewCases',y='Country',orientation = 'h',text='NewCases')



fig_p = px.bar(NewDeaths.sort_values('NewDeaths'), x = 'NewDeaths',y = 'Country',orientation = 'h',text = 'NewDeaths',

              color_discrete_sequence=['#660000'])





fig2 = make_subplots(

    rows=4, cols=2,

    subplot_titles=("ActiveCases","Serious","TotalTests","Tests/1M","NewDeaths",

                   'Deaths/1M pop','NewCases','NewDeaths'))

fig2.add_trace(fig_q['data'][0], row=1, col=1)

fig2.add_trace(fig_r['data'][0], row=1, col=2)

fig2.add_trace(fig_s['data'][0], row=2, col=1)

fig2.add_trace(fig_t['data'][0], row=2, col=2)

fig2.add_trace(fig_u['data'][0], row=3, col=1)

fig2.add_trace(fig_i['data'][0], row=3, col=2)

fig2.add_trace(fig_o['data'][0], row=4, col=1)

fig2.add_trace(fig_p['data'][0], row=4, col=2)



fig2.update_layout(height=3000,title = 'Top 20 countries')



                    

fig2.show()

grp_country1 = time_series.groupby(['Date'])['Confirmed','Deaths','Recovered'].sum().reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Confirmed'],

                    mode='lines',

                    name='Confirmed'))

fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Recovered'],

                    mode='lines',

                    name='Recovered',fillcolor = 'green'))

fig.add_trace(go.Scatter(x=grp_country1['Date'], y=grp_country1['Deaths'],

                    mode='lines',

                    name='Deaths',fillcolor = 'red'))

fig.update_layout(title = 'Confirmed vs Deaths vs Recoverd in world')

fig.show()
grp_country = time_series.groupby(['Date',"Country"])['Confirmed','Deaths','Recovered'].sum().reset_index()



fig_a = px.bar(grp_country, x = 'Date', y = 'Confirmed', color = 'Country',height = 500,

      title = 'Total Confirmed Cases ')



fig_a.show()



fig_b= px.bar(grp_country, x = 'Date', y = 'Deaths',color = 'Country',height = 500,

      title = 'Total Deaths')

fig_b.show()



fig_c=px.bar(grp_country,x='Date',y = 'Recovered', color= 'Country',height = 500,

      title = 'Total Recovered')

fig_c.show()
group1 = time_series.groupby(['Date', 'Country'])['Confirmed', 'Deaths','Recovered'].sum().reset_index()

fig7= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Deaths"]), 

                    hover_name="Country",hover_data = ['Deaths'] ,animation_frame=group1["Date"].dt.strftime('%Y-%m-%d'),

                    projection = 'natural earth',

                    title='Deaths Over Time', color_continuous_scale=px.colors.sequential.Reds)

fig7.update(layout_coloraxis_showscale=False)

fig7.show()






fig8= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Confirmed"]), 

                    hover_name="Country",hover_data = ['Confirmed'] ,animation_frame=group1["Date"].dt.strftime('%Y-%m-%d'),

                    projection = 'natural earth',

                    title='Confirmed Over Time', color_continuous_scale=px.colors.sequential.deep)

fig8.update(layout_coloraxis_showscale=False)

fig8.show()


fig7= px.choropleth(group1, locations="Country", locationmode='country names', color=np.log(group1["Recovered"]), 

                    hover_name="Country",hover_data = ['Recovered'] ,animation_frame=group1["Date"].dt.strftime('%Y-%m-%d'),

                    projection = 'natural earth',

                    title='Recovered Over Time', color_continuous_scale=px.colors.sequential.Greens)

fig7.update(layout_coloraxis_showscale=False)

fig7.show()
HTML('''<div class="flourish-embed" data-src="story/351001" data-url="https://flo.uri.sh/story/351001/embed"><script src=

"https://public.flourish.studio/resources/embed.js"></script></div>''')


usa_df = pd.read_csv('../input/corona-virus-report/usa_county_wise.csv',parse_dates = ['Date'])
usa_df
usa_ = usa_df[usa_df['Date'] == max(usa_df['Date'])]

usa_.head()

usa_grouped = usa_.groupby('Province_State')['Confirmed', 'Deaths'].sum().reset_index()


fig = go.Figure()

fig.add_trace(go.Scatter(x=usa_grouped['Province_State'], y=usa_grouped['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))

fig.add_trace(go.Scatter(x=usa_grouped['Province_State'], y=usa_grouped['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.update_layout(title = 'state wise Confirmed vs Deaths in USA')



fig.show()
usa_grouped1 = usa_df.groupby(['Date'])['Confirmed','Deaths'].sum().reset_index()
fig = go.Figure()

fig.add_trace(go.Scatter(x=usa_grouped1['Date'], y=usa_grouped1['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))

fig.add_trace(go.Scatter(x=usa_grouped1['Date'], y=usa_grouped1['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.update_layout(title = 'Confirmed vs Deaths in USA')

fig.show()
ind_df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',parse_dates = ['Date'])

ind_df.head()
ind_latest = ind_df[ind_df['Date'] == max(ind_df['Date'])]

ind_latest.head()




ind_group = ind_latest.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].sum().reset_index()

ind_group.head()
fig1 = px.bar(ind_group.sort_values('Confirmed'), x='Confirmed', y = 'State/UnionTerritory',orientation = 'h',text = 'Confirmed')

fig1.update_layout(title = 'Confirmed',height = 800)

fig1.show()



fig2 = px.bar(ind_group.sort_values('Deaths'), x='Deaths', y = 'State/UnionTerritory',orientation = 'h',text = 'Deaths',

              color_discrete_sequence=['#F70E0E'])

fig2.update_layout(title = 'Deaths',height = 800)

fig2.show()



fig3 = px.bar(ind_group.sort_values('Cured'), x='Cured', y = 'State/UnionTerritory',orientation = 'h',text = 'Cured',

              color_discrete_sequence=['#F70EFF'])

fig3.update_layout(title='Recovered',height = 800)

fig3.show()