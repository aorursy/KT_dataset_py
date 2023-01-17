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

df_data_6Feb = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv", parse_dates=["Last Update"])

df_data_6Feb["UpdateDate"] = df_data_6Feb["Last Update"].dt.date.astype(str)

df_data2 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv", parse_dates=["Last Update"])

df_data2["UpdateDate"] = df_data2["Last Update"].dt.date.astype(str)

df_data.head()
df_data.describe()
df_data[df_data["Suspected"]>=1].sort_values("Suspected", ascending=False).head()
df_countries = df_data.groupby(['Country/Region', 'Last Update']).sum().reset_index().sort_values('Last Update', ascending=False)

df_countries = df_countries.drop_duplicates(subset = ['Country/Region'])

df_countries = df_countries[df_countries["Confirmed"]>0]

df_countries
data = [ dict(

        type = 'choropleth',

        locations = df_countries['Country/Region'],

        locationmode = 'country names',

        z = df_countries['Confirmed'],

        colorscale=

            [[0.0, "rgb(250, 237, 235)"],

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

      ) ]



layout = dict(

    title = "Last Confirmed Cases (Till January 31, 2020)",

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

                     title="Progression of Coronavirus in Confirmed Cases",template="none")

fig.show()
df_provincebydate = df_data.groupby(['Province/State', 'Last Update', 'UpdateDate']).max().reset_index().sort_values('Last Update', ascending=False)

df_CHProvinces = df_provincebydate[df_provincebydate['Country/Region']=="Mainland China"]

df_chprovincelastcases = df_CHProvinces.drop_duplicates(subset = ['Province/State']).sort_values("Province/State")



df_CHRecDead = df_chprovincelastcases.loc[:,["Province/State", "Recovered", "Death"]]

df_CHRecDeadHb = df_CHRecDead[df_CHRecDead["Province/State"]=="Hubei"]
import seaborn as sns

plt.figure(figsize=(20,8))

sns.barplot(x=df_chprovincelastcases['Province/State'], y = df_chprovincelastcases['Confirmed'])
death = df_CHRecDeadHb['Death'].sum()

recoverd = df_CHRecDeadHb['Recovered'].sum()

import matplotlib.pyplot as plt

labels = 'Death', 'Recoverd'

sizes = [death, recoverd]

colors = 'red', 'green'

explode = [0.1,0]

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=100)

plt.title('death/recovered rate in hubei(wuhan) till january31 2020')

plt.axis('equal')

plt.show()
df_CHRecDeadNotHb = df_CHRecDead[((df_CHRecDead["Province/State"]!="Hubei") & ((df_CHRecDead["Recovered"]>=1) | (df_CHRecDead["Death"]>=1)))].sort_values("Recovered", ascending=False)
df_CHRecDeadNotHb.head()
plt.figure(figsize=(18,8))

sns.barplot(x=df_CHRecDeadNotHb['Province/State'],y=df_CHRecDeadNotHb['Recovered'], color='blue',label='RECOVERED')

sns.barplot(x=df_CHRecDeadNotHb['Province/State'], y=df_CHRecDeadNotHb['Death'], color='red', label='DEATH')

plt.title('death recoverd rate in other china provinces')

plt.legend()
df_CHProvincesByDate = df_CHProvinces.groupby(['Province/State', 'UpdateDate']).max().reset_index().sort_values('UpdateDate') 

df_CHProvincesByDateHB = df_CHProvincesByDate[df_CHProvincesByDate["Province/State"]=="Hubei"]
#confirmed = df_CHRecDeadHb['Confirmed']

plt.figure(figsize=(15,5))

sns.lineplot(x=df_CHProvincesByDateHB["UpdateDate"],y=df_CHProvincesByDateHB['Confirmed'])

plt.title('confirmed cases in hubei')
df_CHProvincesByDateHB.set_index('UpdateDate')
Data = df_CHProvincesByDateHB.iloc[:,-2:]
plt.figure(figsize=(10,5))

sns.lineplot(data=Data, markers=True,)

plt.title('recovered/deaths')
df_6feb = df_data_6Feb.groupby(['Country/Region', 'Last Update']).sum().reset_index().sort_values('Last Update', ascending=False)

df_6feb = df_6feb.drop_duplicates(subset = ['Country/Region'])

df_6feb = df_6feb[df_6feb["Confirmed"]>0]

df_6febNotCh = df_6feb[df_6feb['Country/Region']!='Mainland China']

df_6febNotCh = df_6febNotCh.sort_values("Confirmed")
plt.figure(figsize=(10,10))

sns.barplot(x=df_6febNotCh['Confirmed'], y = df_6febNotCh['Country/Region'])

plt.title('no. of cases outside mainland china until 6feb')