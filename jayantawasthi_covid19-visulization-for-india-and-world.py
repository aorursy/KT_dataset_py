# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid=pd.read_csv("/kaggle/input/covid19-datasetoctober/owid-covid-data.csv")
covid.head()
covid.info()
covid_india=covid[covid["location"]=="India"]
covid_india.info()
covid_india.isnull().sum()
covid_india.head()
covid_india=pd.DataFrame(covid_india)
covid_india["date"]=pd.to_datetime(covid_india["date"],infer_datetime_format=True)
covid_india=covid_india.set_index(['date'])
covid_india.head()
covid_india.drop(["iso_code","continent","location"],axis=1,inplace=True)
covid_india.head()
def plo(j,m,k):
            fig = px.line(j,y=m)
            fig.update_layout(
            title={'text':k,'x':0.5},title_font_color="black")
            fig.update_xaxes(rangeslider_visible=True)
            fig.show()
plo(covid_india["new_cases"],"new_cases","New Cases in India")
plo(covid_india["total_cases"],"total_cases","total cases in India")
plo(covid_india["new_cases_smoothed"],"new_cases_smoothed","new_cases_smoothed")
plo(covid_india["total_deaths"],"total_deaths","Total_Deaths")
plo(covid_india["new_deaths"],"new_deaths","Daily_Deaths")
plo(covid_india["total_cases_per_million"],"total_cases_per_million","Total_Cases_Per_Million")
plo(covid_india["total_deaths_per_million"],"total_deaths_per_million","Total Deaths Per Million")
plo(covid_india["gdp_per_capita"],"gdp_per_capita","Gdp_Per_Capita")
covid.head()
rolmean=covid_india["total_cases"].rolling(window=7).mean()
def rolme(j,k,l,b):
             rolmean=j.rolling(window=7).mean()
             fig = px.line(x=covid_india.index, y=[j,rolmean])
             fig.update_layout(
             title={'text':k,'x':0.5},title_font_color="black")
             fig.update_xaxes(rangeslider_visible=True)
             fig.show()
rolme(covid_india["new_cases"],"New Case vs Rolling Mean for last 7 days","new_cases","rol_mean for window size of 7")
rolme(covid_india["new_deaths"],"New Deaths vs Rolling Mean for last 7 days","new_deaths","rol_mean for window size of 7")
covid.head()
t=(covid[covid["date"]=="2020-10-14"].index.values)
locat=[]
cases=[]
def cov(th,thh):
    for i in th:
        if covid["total_cases"].iloc[i]>thh:
                                 tt=covid["total_cases"].iloc[i]
                                 ttt=covid["location"].iloc[i]
                                 cases.append(tt)
                                 locat.append(ttt)
              
cov(t,500000)
locat
locat.pop()
cases.pop()
fig = px.bar(x=cases, y=locat, orientation='h',title="Top 13 Covid Countries")
fig.show()
locat=[]
cases=[]
cov(t,0)
locat.pop()
locat.pop()
cases.pop()
cases.pop()
fig = px.pie(values=cases, names=locat, title='Country Shares in Covid cases')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
covid_usa=covid[covid["location"]=="United States"]
covid_brazil=covid[covid["location"]=="Brazil"]
def ivso(i,k):
    r1=covid_usa[i].rolling(window=7).mean()
    r2=covid_brazil[i].rolling(window=7).mean()
    r3=covid_india[i].rolling(window=7).mean()
    fig = px.line(x=covid_india.index, y=[r1,r2,r3])
    fig.update_layout(
    title={'text':k,'x':0.5},title_font_color="black")
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
ivso("new_cases","Moving Average of New_Cases of top 4 Covid Countries")
ivso("total_cases","MA of Total_Cases of top 3 Covid Nations")
ivso("total_deaths","MA of total_deaths of top 3 covid Nations")
ivso("new_deaths","MA of New_Deaths in top 3 Covid Nations")
ivso("total_cases_per_million","MA of total_cases_per_million in top 3 covid nations")
ivso("new_cases_per_million","MA of new_cases_per_million in top 3 covid Nations")
a=[]
for i in range(50512):
      if covid["location"].iloc[i]=='World':
                       a.append(i)
covid.drop(a,inplace=True)
len(covid)
covid["date"]=pd.to_datetime(covid["date"])
covid=covid.sort_values(by='date')
coc=covid.copy()
coc.to_csv("coca.csv")
coca=pd.read_csv("coca.csv")
coca.head()
coca.drop("Unnamed: 0",axis=1,inplace=True)
len(coca)
fig=px.choropleth(coca,locations="iso_code",color="total_cases",hover_name="location",animation_frame=coca["date"],color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(transition={'duration':1000})
fig.show()