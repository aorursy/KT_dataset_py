# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import folium

import math

import random

from datetime import timedelta



import warnings

warnings.filterwarnings("ignore")



#Color Palletes



cnf = '#393e46'

dth = '#ff2e63'

rec = '#21bf73'

act = '#fe9801'



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv",parse_dates=['Date'])
covid_df.head()
covid_df["Province/State"] = covid_df["Province/State"].fillna("")

covid_df
country_daywise =pd.read_csv("/kaggle/input/corona-virus-report/full_grouped.csv",parse_dates=['Date'])

countrywise =pd.read_csv("/kaggle/input/corona-virus-report/country_wise_latest.csv")

daywise =pd.read_csv("/kaggle/input/corona-virus-report/day_wise.csv",parse_dates=['Date'])
confirmed = covid_df.groupby('Date').sum()['Confirmed'].reset_index()

confirmed
recovered = covid_df.groupby('Date').sum()['Recovered'].reset_index()

recovered
deaths = covid_df.groupby('Date').sum()['Deaths'].reset_index()

deaths
covid_df.isnull().sum()
covid_df.info()
fig =go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'],y =confirmed['Confirmed'], mode='lines+markers',name= 'Confirmed',line=dict(color="Orange",width=2)))

fig.add_trace(go.Scatter(x=recovered['Date'],y =recovered['Recovered'], mode='lines+markers',name= 'Recovered',line=dict(color="Green",width=2)))

fig.add_trace(go.Scatter(x=confirmed['Date'],y =deaths['Deaths'], mode='lines+markers',name= 'Deaths',line=dict(color="Red",width=2)))

fig.update_layout(title='Worldwide Covid-19 cases',xaxis_tickfont_size =14,yaxis=dict(title="No of cases"))

fig.show()
covid_df["Date"]=covid_df["Date"].astype(str)
covid_df.info()
fig = px.density_mapbox(covid_df,lat = "Lat",lon="Long",hover_name="Country/Region",hover_data=['Confirmed','Recovered','Deaths'],animation_frame='Date',color_continuous_scale='Portland',radius=7,zoom=0,height=700)

fig.update_layout(title='WorldWide Covid 19 cases with Time lapse')

fig.update_layout(mapbox_style= "open-street-map",mapbox_center_lon=0)

fig.show()
temp = covid_df.groupby('Date')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop= True)

tm = temp.melt(id_vars='Date',value_vars=['Active','Deaths','Recovered'])

fig = px.treemap(tm,path=["variable"],values='value',height=250,width = 800,color_discrete_sequence=[act,rec,dth])

fig.data[0].textinfo ='label+text+value'

fig.show()
temp = covid_df.groupby('Date')["Recovered","Deaths","Active"].sum().reset_index()

temp = temp.melt(id_vars= "Date",value_vars= ["Recovered","Deaths","Active"],var_name='Case',value_name="Count")

fig = px.area(temp,x="Date",y="Count",color="Case",height=400,title="Cases Over time",color_discrete_sequence=[rec,dth,act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
temp = covid_df[covid_df["Date"]==max(covid_df["Date"])]

m = folium.Map(location =[0,0],titles = 'cartodbpositron',min_zoom=1, max_zomm=4,zoom_start=1)

for i in range(0,len(temp)):

    folium.Circle(location=[temp.iloc[i]['Lat'],temp.iloc[i]['Long']],color= 'crimson',fill = 'crimson',

                  tooltip='<li><bold> Country: ' + str(temp.iloc[i]['Country/Region'])+

                           '<li><bold> Province: ' + str(temp.iloc[i]['Province/State'])+

                            '<li><bold> Confirmed Cases: ' + str(temp.iloc[i]['Confirmed'])+

                            '<li><bold> Deaths Cases: ' + str(temp.iloc[i]['Deaths']),

                  radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)

m
fig = px.choropleth(country_daywise,locations='Country/Region',locationmode='country names',color=np.log(country_daywise['Confirmed']),

                   hover_name='Country/Region',animation_frame=country_daywise['Date'].dt.strftime('%Y-%m-%d'),

                   title='Cases over time',color_continuous_scale=px.colors.sequential.Inferno)

fig.update(layout_coloraxis_showscale= True)

fig.show()
daywise.head()
fig_c = px.bar(daywise,x='Date',y = 'Confirmed',color_discrete_sequence=[act])

fig_d = px.bar(daywise,x='Date',y = 'Deaths',color_discrete_sequence=[dth])



fig = make_subplots(rows=1,cols=2,shared_xaxes=False,horizontal_spacing=0.1,

                   subplot_titles=('Confirmed Cases','Death Cases'))

fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)



fig.update_layout(height=400)

fig.show()
fig_c= px.choropleth(countrywise,locations='Country/Region',locationmode='country names',

                    color=np.log(countrywise['Confirmed']),hover_name="Country/Region",

                    hover_data=['Confirmed'])

temp= countrywise[countrywise["Deaths"]>0]

fig_d= px.choropleth(temp,locations='Country/Region',locationmode='country names',

                    color=np.log(temp['Deaths']),hover_name="Country/Region",

                    hover_data=['Deaths'])

fig = make_subplots(rows=1,cols=2,subplot_titles=['Confirmed','Deaths'],

                   specs=[[{'type':'choropleth'},{'type':'choropleth'}]])



fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_c['data'][0],row=1,col=2)

fig.update(layout_coloraxis_showscale=False)

fig.show()
daywise.columns
fig1=px.line(daywise,x='Date',y='Deaths / 100 Cases',color_discrete_sequence=[dth])

fig2=px.line(daywise,x='Date',y='Recovered / 100 Cases',color_discrete_sequence=[rec])

fig3=px.line(daywise,x='Date',y='Deaths / 100 Recovered',color_discrete_sequence=[rec])



fig=make_subplots(rows=1,cols=3,shared_xaxes=False,

                 subplot_titles=('Deaths / 100 Cases','Recovered / 100 Cases','Deaths / 100 Recovered'))



fig.add_trace(fig1['data'][0],row=1,col=1)

fig.add_trace(fig2['data'][0],row=1,col=2)

fig.add_trace(fig3['data'][0],row=1,col=3)



fig.update_layout(height=400)

fig.show()
fig_c=px.bar(daywise,x='Date',y='Confirmed',color_discrete_sequence=[act])

fig_d=px.bar(daywise,x='Date',y='No. of countries',color_discrete_sequence=[dth])



fig= make_subplots(rows=1,cols=2,shared_xaxes=False,horizontal_spacing=0.1,

                  subplot_titles=('No of new cases per day','No of countries'))



fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)



fig.show()
countrywise.head()
top =15



fig_c=px.bar(countrywise.sort_values('Confirmed').tail(top),x='Confirmed',y="Country/Region",

            text='Confirmed',orientation='h',color_discrete_sequence=[act])

fig_d=px.bar(countrywise.sort_values('Deaths').tail(top),x='Deaths',y="Country/Region",

            text='Deaths',orientation='h',color_discrete_sequence=[dth])



fig_a=px.bar(countrywise.sort_values('Active').tail(top),x='Active',y="Country/Region",

            text='Active',orientation='h',color_discrete_sequence=['#434343'])

fig_r=px.bar(countrywise.sort_values('Recovered').tail(top),x='Recovered',y="Country/Region",

            text='Recovered',orientation='h',color_discrete_sequence=[rec])



fig_dc=px.bar(countrywise.sort_values('Deaths / 100 Cases').tail(top),x='Deaths / 100 Cases',y="Country/Region",

            text='Deaths / 100 Cases',orientation='h',color_discrete_sequence=['#f84351'])

fig_rc=px.bar(countrywise.sort_values('Recovered / 100 Cases').tail(top),x='Recovered / 100 Cases',y="Country/Region",

            text='Recovered / 100 Cases',orientation='h',color_discrete_sequence=['#a45398'])



fig_nc=px.bar(countrywise.sort_values('New cases').tail(top),x='New cases',y="Country/Region",

            text='New cases',orientation='h',color_discrete_sequence=['#f04341'])



fig_wc=px.bar(countrywise.sort_values('1 week change').tail(top),x='1 week change',y="Country/Region",

            text='1 week change',orientation='h',color_discrete_sequence=['#c04341'])

temp = countrywise[countrywise['Confirmed'] > 100]

fig_wi=px.bar(temp.sort_values('1 week % increase').tail(top),x='1 week % increase',y="Country/Region",

            text='1 week % increase',orientation='h',color_discrete_sequence=['#b05398'])



fig=make_subplots(rows=5,cols=2,shared_xaxes=False,horizontal_spacing=0.14,

                 vertical_spacing=0.1,subplot_titles=('Confirmed Cases','Deaths Reported','Active Cases',

                                                      'Recovered Cases','Deaths / 100 Cases','Recovered / 100 Cases',

                                                      'New Cases','Cases / Million People',

                                                     '1 week % increase'))



fig.add_trace(fig_c['data'][0],row=1,col=1)

fig.add_trace(fig_d['data'][0],row=1,col=2)

fig.add_trace(fig_a['data'][0],row=2,col=1)

fig.add_trace(fig_r['data'][0],row=2,col=2)



fig.add_trace(fig_dc['data'][0],row=3,col=1)

fig.add_trace(fig_rc['data'][0],row=3,col=2)



fig.add_trace(fig_nc['data'][0],row=4,col=1)

fig.add_trace(fig_wc['data'][0],row=4,col=2)



fig.add_trace(fig_wi['data'][0],row=5,col=1)



fig.update_layout(height=3000)

fig.show()
top=15

fig = px.scatter(countrywise.sort_values('Deaths',ascending=False).head(top),

                x= 'Confirmed',y='Deaths',color='Country/Region',size = 'Confirmed',height=600,

                text='Country/Region',log_x=True,log_y=True,title ='Deaths vs Confirmed Cases (Cases on log scales)')

fig.update_traces(textposition='top center')

fig.update_layout(showlegend= True)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig=px.bar(country_daywise,x='Date',y='Confirmed',color='Country/Region',height=600,

          title='Confirmed Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig=px.bar(country_daywise,x='Date',y='Deaths',color='Country/Region',height=600,

          title='Deaths Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig=px.bar(country_daywise,x='Date',y='Recovered',color='Country/Region',height=600,

          title='Recovered Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig=px.bar(country_daywise,x='Date',y='New cases',color='Country/Region',height=600,

          title='New cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig =px.line(country_daywise,x='Date',y='Confirmed',color='Country/Region',height=600,

            title='Confirmed',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig =px.line(country_daywise,x='Date',y='Deaths',color='Country/Region',height=600,

            title='Deaths cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig =px.line(country_daywise,x='Date',y='Recovered',color='Country/Region',height=600,

            title='Recovered cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
full_latest= covid_df[covid_df['Date'] == max(covid_df['Date'])]

fig = px.treemap(full_latest.sort_values(by='Confirmed',ascending=False).reset_index(drop=True),

                path=['Country/Region','Province/State'],values='Confirmed',height=700,

                title='Number of Confirmed Cases',

                color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()
full_latest= covid_df[covid_df['Date'] == max(covid_df['Date'])]

fig = px.treemap(full_latest.sort_values(by='Deaths',ascending=False).reset_index(drop=True),

                path=['Country/Region','Province/State'],values='Deaths',height=700,

                title='Number of Deaths Cases',

                color_discrete_sequence=px.colors.qualitative.Dark2)

fig.data[0].textinfo='label+text+value'

fig.show()
#Wikipedia Source

epidemics = pd.DataFrame({

    'epidemic': ['COVID-19','SARS','EBOLA','MERS','H1N1'],

    'start_year': [2019,2002,2013,2012,2009],

    'end_year': [2020,2004,2016,2020,2010],

    'confirmed' : [full_latest['Confirmed'].sum(),8422,28646,2519,6724149],

    'deaths' : [full_latest['Deaths'].sum(),813,11323,866,19654]

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100,2)

epidemics.head()
temp = epidemics.melt(id_vars='epidemic',value_vars=['confirmed','deaths','mortality'],

                     var_name='Case',value_name='Value')

fig=px.bar(temp,x='epidemic',y='Value',color='epidemic',text='Value',facet_col='Case',

          color_discrete_sequence=px.colors.qualitative.Bold)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8,uniformtext_mode='hide')

fig.update_yaxes(showticklabels=False)

fig.layout.yaxis2.update(matches=None)

fig.layout.yaxis3.update(matches=None)

fig.show()