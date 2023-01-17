import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')
df_data = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv", parse_dates=["Last Update"])

df_data["UpdateDate"] = df_data["Last Update"].dt.date.astype(str)

df_data2 = pd.read_csv("../input/coronavirus-disease-covid19-dataset/2020_nCoV_data.csv", parse_dates=["Last Update", "Date"])

df_data2["Country"] = df_data2["Country"].str.replace("Mainland China", "China")

df_data.head()
df_data.describe().T
df_data[df_data["Suspected"]>=1].sort_values("Suspected", ascending=False).head()
df_countries = df_data2.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)

df_countries = df_countries.drop_duplicates(subset = ['Country'])

df_countriesConf = df_countries[df_countries["Confirmed"]>0]
df_countriesConf
data = [ dict(

        type = 'choropleth',

        locations = df_countriesConf['Country'],

        locationmode = 'country names',

        z = df_countriesConf['Confirmed'],

        colorscale=

            [[0.0, "rgb(251, 237, 235)"],

            [0.09, "rgb(245, 211, 206)"],

            [0.12, "rgb(239, 179, 171)"],

            [0.15, "rgb(236, 148, 136)"],

            [0.22, "rgb(239, 117, 100)"],

            [0.35, "rgb(235, 90, 70)"],

            [0.45, "rgb(207, 81, 61)"],

            [0.65, "rgb(176, 70, 50)"],

            [0.85, "rgb(147, 59, 39)"],

            [1.00, "rgb(110, 47, 26)"]],

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) 

        ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Participant'),

) 

       ]



layout = dict(

    title = "Last Confirmed Cases (Till February 17, 2020)",

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(type = 'Mercator'),

        width=500,height=400)

)



w_map = dict( data=data, layout=layout)

iplot( w_map, validate=False)
df_countrybydate = df_data.groupby(['Country/Region', 'Last Update', 'UpdateDate']).sum().reset_index().sort_values('Last Update', ascending=False)

df_countrybydate = df_countrybydate.groupby(['Country/Region', 'UpdateDate']).max().reset_index().sort_values('Last Update')

df_countrybydate["Size"] = np.where(df_countrybydate['Country/Region']=='Mainland China', df_countrybydate['Confirmed'], df_countrybydate['Confirmed']*200)
df = px.data.gapminder()

fig = px.scatter_geo(df_countrybydate, locations="Country/Region", locationmode = "country names",

                     hover_name="Country/Region", size="Size", color="Confirmed",

                     animation_frame="UpdateDate", 

                     projection="natural earth",

                     title="Progression of Coronavirus in Confirmed Cases in the January 2020",template="none")

fig.show()
df_provincebydate = df_data2.groupby(['Province/State', 'Date']).max().reset_index().sort_values('Date', ascending=False)

df_CHProvinces = df_provincebydate[df_provincebydate['Country']=="China"]

df_CHProvinces = df_CHProvinces.drop_duplicates(subset = ['Province/State']).sort_values("Province/State")

df_CHProvinces = df_CHProvinces[~df_CHProvinces['Province/State'].isin(['Macau', 'Taiwan'])]

#I excluded 'Macau', 'Taiwan' from China because they have own countries in the dataset.

df_CHProvinces = df_CHProvinces[df_CHProvinces["Confirmed"]>0]



df_CHRecDead = df_CHProvinces.loc[:,["Province/State", "Recovered", "Deaths"]]

df_CHRecDeadHb = df_CHRecDead[df_CHRecDead["Province/State"]=="Hubei"]
fig = go.Figure()



fig.add_trace(go.Bar(

                x=df_CHProvinces["Province/State"],

                y=df_CHProvinces["Confirmed"],

                marker_color='darkorange',

                marker_line_color='rgb(8,48,107)',

                marker_line_width=2, 

                opacity=0.7)

             )



fig.update_layout(

    title_text='Confirmed Cases on Provinces of China (Till February 17, 2020)',

    height=700, width=800, xaxis_title='Province/State', yaxis_title='Confirmed')



fig.show()
colors = ['mediumturquoise', 'orangered']

columns = list(df_CHRecDeadHb.iloc[:,1:3])

values = df_CHRecDeadHb.iloc[:,1:3].values.tolist()[0]



fig = go.Figure(data=[go.Pie(labels=columns, 

                             values=values , hole=.3)]

               )



fig.update_traces(hoverinfo='label+percent+value', textinfo='label+percent', textfont_size=18,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2))

                 )



fig.update_layout(

    title_text="Death/Recovered Rate in Hubei (Wuhan) (Till February 17, 2020)", height=500, width=700, showlegend=False

)



fig.show()
df_CHRecDeadNotHb = df_CHRecDead[((df_CHRecDead["Province/State"]!="Hubei") & ((df_CHRecDead["Recovered"]>=1) | (df_CHRecDead["Deaths"]>=1)))].sort_values("Recovered", ascending=False)
fig = go.Figure()



fig.add_trace(go.Bar(

                x=df_CHRecDeadNotHb["Province/State"],

                y=df_CHRecDeadNotHb["Recovered"],

                marker_color='mediumturquoise',

                name="Recovered")

             )



fig.add_trace(go.Bar(

                x=df_CHRecDeadNotHb["Province/State"],

                y=df_CHRecDeadNotHb["Deaths"],

                marker_color='red',

                name="Deaths")

             )



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)



fig.update_layout(

    title_text='Death/Recovered Rate in the Other China Provinces Except Wuhan',

    height=600, width=800, xaxis_title='Province/State'

)



fig.show()
df_CHProvincesByDateHB = df_provincebydate[df_provincebydate['Province/State']=="Hubei"]



fig = go.Figure()



fig.add_trace(go.Scatter(x=df_CHProvincesByDateHB["Date"], y=df_CHProvincesByDateHB["Confirmed"],

                         line=dict(color='indigo', width=3), name="Confirmed")

             )



fig.update_layout(title='Confirmed Cases By Date in Hubei',

                   xaxis_title='Date',

                   yaxis_title='Count',

                   width=740, height=350)



fig.show()





fig = go.Figure()



fig.add_trace(go.Scatter(x=df_CHProvincesByDateHB["Date"], y=df_CHProvincesByDateHB["Deaths"],

                         line=dict(color='crimson', width=3), name="Deaths"))





fig.add_trace(go.Scatter(x=df_CHProvincesByDateHB["Date"], y=df_CHProvincesByDateHB["Recovered"],

                         line=dict(color='darkcyan', width=3), name="Recovered"))



fig.update_layout(title='Recovery and Death Cases By Date in Hubei',

                   xaxis_title='Date',

                   yaxis_title='Count',

                   width=800, height=350)



fig.show()
df_countriesNotCh = df_countries[~df_countries['Country'].isin(['China', 'Others'])]

df_countriesNotChConf = df_countriesNotCh[df_countriesNotCh["Confirmed"]>0]

df_countriesNotChConf = df_countriesNotChConf.sort_values("Confirmed")
fig = go.Figure()

fig.add_trace(go.Bar(x=df_countriesNotChConf["Confirmed"],

                y=df_countriesNotChConf['Country'],

                marker_color='powderblue',

                marker_line_color='rgb(8,48,107)',

                marker_line_width=1.5, 

                opacity=0.6,

                orientation='h'))



fig.update_layout(

    title_text='Last Confirmed Cases Outside of China (Till February 17, 2020)',

    height=700, width=800,

    showlegend=False, xaxis_title='Confirmed Cases') 



fig.show()
df_countriesNotChRec = df_countriesNotCh[df_countriesNotCh["Recovered"]>0]

df_countriesNotChRec = df_countriesNotChRec.sort_values("Recovered")



df_countriesNotChDeath = df_countriesNotCh[df_countriesNotCh["Deaths"]>0]

df_countriesNotChDeath = df_countriesNotChDeath.sort_values("Deaths")
fig = go.Figure()



fig.add_trace(go.Bar(x=df_countriesNotChRec["Recovered"],

                y=df_countriesNotChRec['Country'],

                marker_color='mediumseagreen',

                name="Recovered",

                orientation='h'))



fig.add_trace(go.Bar(x=df_countriesNotChRec["Deaths"],

                y=df_countriesNotChRec['Country'],

                marker_color='maroon',

                name="Deaths",

                orientation='h'))



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)





fig.update_layout(

    title_text='Death/Recovered Cases Outside of China (Till February 17, 2020)',

    height=700, width=800,

    xaxis_title='Death/Recovered Cases') 



fig.show()