# import libraries



import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
# using pandas read_csv

dataset ='/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv'

df = pd.read_csv(dataset, parse_dates=['Date'])
df.tail()
# On Jan.22,2020, row 0 was marked as "China", now change it to "Mainland China".

df.loc[df.Country == 'China', 'Country'] = 'Mainland China'
df.loc[df.Country == 'Others', 'Country'] = 'Japan'# Others =Princess Diamond Cruise Ship
# plot confirmed cases outside mainland China based on countries



import plotly.express as px

# filter data based on the last update

df2=df[(df['Country']!= 'Mainland China') & (df['Date']=='2020-02-11 20:44:00')]



df3=df2.groupby(['Country'],as_index=False).sum().sort_values(by=['Confirmed'], ascending=False)

fig = px.bar(df3, x='Country', y='Confirmed',

             color='Confirmed', height=400)

fig.update_layout(title_text='Coronavirus confirmed cases outside mainland China')

fig.show()
#!pip install chart-studio
"""

import chart_studio.plotly as py

import cufflinks as cf

df3.iplot(kind='bubble', x='Country', y='Confirmed', size='Confirmed', text='Country',

             xTitle='Confirmed cases outside mainland China', yTitle='Confirmed',

             filename='cufflinks/simple-bubble-chart'



"""

#Because the above cell code does not work on Kaggle, I imported the plotted picture. 

from IPython.display import Image

Image("/kaggle/input/firstonep/first.png")
# Now I want to compare the trend of two areas in and outside mainland China

#all the other regions/countries grouped as outside (mainland)China



df.loc[df.Country != 'Mainland China', 'Country'] = 'OutsideChina'
df4=df.groupby(['Date','Country'],as_index=False).sum().sort_values(by=['Confirmed'], ascending=False)
#removed redundant column "Sno(Serial Number)"



df5=df4[['Date','Country','Confirmed','Deaths','Recovered']]
#subset df2 (in Mainland China and Outside mainland china--> prepare for visulaization)

dfChina=df5[df5['Country']== 'Mainland China'] 
import plotly.graph_objects as go

fig = go.Figure()



    

fig.add_trace(go.Scatter(

                x=dfChina['Date'],

                y=dfChina['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=dfChina['Date'],

                y=dfChina['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=dfChina['Date'],

                y=dfChina['Recovered'],

                name="Recovered",

                line_color='dimgray',

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(title_text='Coronavirus Outbreak in Mainland China with Rangeslider',

                  xaxis_rangeslider_visible=True)

fig.show()
dfOutChina=df5[df5['Country']== 'OutsideChina'] 
import plotly.express as px #based on date



fig = px.bar(dfOutChina, x='Date', y='Confirmed',

             color='Confirmed', height=400)

fig.update_layout(title_text='Coronavirus confirmed cases outside mainland China')

fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(name='Confirmed', x=dfOutChina['Date'], y=dfOutChina['Confirmed']))

fig.add_trace(go.Bar(name='Deaths', x=dfOutChina['Date'], y=dfOutChina['Deaths']))

fig.add_trace(go.Bar(name='Recovered', x=dfOutChina['Date'], y=dfOutChina['Recovered']))

fig.update_layout(title_text='Coronavirus Outbreak outside Mainland China with Rangeslider',

                  xaxis_rangeslider_visible=True, xaxis_tickangle=-15,barmode='group')

fig.show()
"""

##This is the second one that does not work on Kaggle. You can see the plotted picture under this cell



dfOutChina.iplot(kind='scatter', mode='markers', x='Date', y='Confirmed', filename='cufflinks/simple-scatter')

"""
#this is another picture which does not show on Kaggle



from IPython.display import Image

Image("/kaggle/input/lastonep/second.png")
#The next, I would like to show the outbreak situation in mainland China. 

# The first, let's have a look at the confirmed cases. I will removed countries and regions outside mainland China. Then, I will 

#make a compariation between two groups: the outbreak center Hubei province and the other provinces.  

dataset ='/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv'

dftimeCfm = pd.read_csv(dataset)

dftimeCfm.head()

subset=dftimeCfm[dftimeCfm['Country/Region']== 'Mainland China'] 

subset.tail()
#now we have a subset that contains only date, confirmed cases in Hubei and other provinces

import warnings

warnings.filterwarnings('ignore')

subset.drop(['Lat','Long'], axis=1, inplace=True)

subset["Province/State"]=np.where(subset["Province/State"].eq("Hubei"), "Hubei", "Other")

subsetCfmd=subset.groupby(['Province/State'],as_index=False).sum()

new=subsetCfmd.set_index('Province/State').T.reset_index()

new.columns = ['Date', 'HubeiCfmd', 'OtherCfmd']

new.head()
#plot confirmed cases in mainland China

fig = go.Figure()

fig.add_trace(go.Scatter(x=new['Date'], y=new['HubeiCfmd'],

                    mode='lines+markers',

                    name='Hubei Province',

                    line_color='Red'))

fig.add_trace(go.Scatter(x=new['Date'], y=new['OtherCfmd'],

                    mode='lines+markers',

                    name='Other Provinces',

                    line_color="Green"))

fig.update_layout(title_text='Confirmed cases in mainland China: Hubei Province VS Other Provinces')



fig.show()
# We will do the same to compare deaths and recovered cases

dataset ='/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv'

dftimeDeaths = pd.read_csv(dataset)

dftimeDeaths.head()

subset2=dftimeDeaths[dftimeDeaths['Country/Region']== 'Mainland China'] 

subset2.drop(['Lat','Long'], axis=1, inplace=True)

subset2["Province/State"]=np.where(subset2["Province/State"].eq("Hubei"), "Hubei", "Other")

subDeaths=subset2.groupby(['Province/State'],as_index=False).sum()

new2=subDeaths.set_index('Province/State').T.reset_index()

new2.columns = ['Date', 'HubeiDeaths', 'OtherDeaths']

new2.head()

fig = go.Figure()

fig.add_trace(go.Scatter(x=new2['Date'], y=new2['HubeiDeaths'],

                    line=dict(color='firebrick', width=5,dash='dot'),

                    name='Hubei Province'))

fig.add_trace(go.Scatter(x=new2['Date'], y=new2['OtherDeaths'],

                    line=dict(color='blue', width=5,dash='dot'),

                    name='Other Provinces'))

fig.update_layout(title_text='Deaths cases in mainland China: Hubei Province VS Other Provinces')



fig.show()
dataset ='/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv'

dftimeRec = pd.read_csv(dataset)

dftimeRec.head()

subset3=dftimeRec[dftimeRec['Country/Region']== 'Mainland China'] 

subset3.drop(['Lat','Long'], axis=1, inplace=True)

subset3["Province/State"]=np.where(subset3["Province/State"].eq("Hubei"), "Hubei", "Other")

subRec=subset3.groupby(['Province/State'],as_index=False).sum()

new3=subRec.set_index('Province/State').T.reset_index()

new3.columns = ['Date', 'HubeiRecovered', 'OtherRecovered']

new3.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=new3['Date'], y=new3['HubeiRecovered'],

                    line=dict(color='Orange', width=5,dash='dash'),

                    name='Hubei Province'))

fig.add_trace(go.Scatter(x=new3['Date'], y=new3['OtherRecovered'],

                    line=dict(color='Brown', width=5,dash='dash'),

                    name='Other Provinces'))

fig.update_layout(title_text='Recovered cases in mainland China: Hubei Province VS Other Provinces')



fig.show()
fig = go.Figure()







fig.add_trace(go.Scatter(x=new['Date'], y=new['HubeiCfmd'],

                    mode='lines+markers',

                   name='Confirmed',

                    line_color='Red'))



fig.add_trace(go.Scatter(x=new2['Date'], y=new2['HubeiDeaths'],

                    line=dict(color='firebrick', width=5,dash='dot'),

                    name='Deaths'))



fig.add_trace(go.Scatter(x=new3['Date'], y=new3['HubeiRecovered'],

                    line=dict(color='Orange', width=5,dash='dash'),

                    name='Recovered'))



fig.update_layout(title_text='Hubei Province, center of the coronavirus outbreak')



fig.show()
#let's sum up those subsets and make some pies based on confirmed cases //I will rewrite the code

newSum=new[new['Date'] =='02/09/20 23:20']

newSum=newSum.set_index('Date').T.reset_index()

newSum.columns = [ 'Province', 'Cases']

newSum.head()
df = px.data.tips()

colors=['green','Dark orange']

fig = px.pie(newSum, values='Cases', names='Province')

fig.update_traces(textposition='inside', textinfo='percent+label',marker=dict(colors=colors))



fig.update_layout(title_text='Confirmed cases by Feb.09,2020 in Hubei Province compared with the other provinces')

fig.show()
#let's sum up those subsets and make some pies based on deaths cases in mainland China//I will rewrite the code

new2Sum=new2[new2['Date'] =='02/09/20 23:20']

new2Sum=new2Sum.set_index('Date').T.reset_index()

new2Sum.columns = [ 'Province', 'Cases']

new2Sum.head()

df = px.data.tips()

colors = ['gold', 'mediumturquoise']

fig = px.pie(new2Sum, values='Cases', names='Province')



fig.update_traces(textposition='inside', textinfo='percent+label',marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text='Deaths cases by Feb.09,2020 in Hubei Province compared with the other provinces')

fig.show()
#let's sum up those subsets and make some pies based on recovered cases in mainland China//I will rewrite the code

new3Sum=new3[new3['Date'] =='02/09/20 23:20']

new3Sum=new3Sum.set_index('Date').T.reset_index()

new3Sum.columns = [ 'Province', 'Cases']

new3Sum.head()

df = px.data.tips()



labels = new3Sum['Province']

values=new3Sum['Cases']



fig = go.Figure(data=[go.Pie(labels=labels,values=values, hole=.3)])

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(title_text='Recovered cases by Feb.09,2020 in Hubei Province compared with the other provinces')

fig.show()