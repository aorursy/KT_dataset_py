# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.io as pio

import plotly.graph_objects as go

import missingno as  msno

from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/malaria-dataset/reported_numbers.csv")

df.head().style.background_gradient(cmap='Reds')
df.info()
df.isnull().sum()
n = msno.bar(df,color = "lightskyblue")
df.dropna(inplace=True)

df.isnull().sum()
df_group =df.groupby('Country')["No. of cases","No. of deaths"].sum().reset_index()

df_group.head()
df_cases = df_group[["Country","No. of cases"]]

df_cases.head()
pio.templates.default ='plotly_dark'

fig = px.bar(df_cases.sort_values("No. of cases",ascending=False)[:20][::-1],x="No. of cases",y ="Country",text="No. of cases",

             title="Top 20 Country with highest number of Malaria Cases till 2000 to 2018",

             color_discrete_sequence= px.colors.qualitative.Light24,height=900,orientation="h")#

fig.show()
df_death = df_group[["Country","No. of deaths"]]

pio.templates.default ='plotly_dark'

fig = px.bar(df_death.sort_values("No. of deaths",ascending=False)[:20][::-1],x="No. of deaths",y ="Country",text="No. of deaths",

             title="Top 20 Country with highest number of Malaria Deaths from 2000 to 2018",

             color_discrete_sequence= px.colors.qualitative.Light24,height=800,orientation="h")

fig.show()



who_group =df.groupby('WHO Region')["No. of cases","No. of deaths"].sum().reset_index()

who_group.head().style.background_gradient(cmap ='Reds')
pio.templates.default = "plotly_dark"

fig = px.bar(who_group.sort_values("No. of cases",ascending=False)[::-1],y="No. of cases",x ="WHO Region",text="No. of cases",

             title="WHO regions with highest number of Cases from 2000 to 2018",

             color_discrete_sequence= px.colors.qualitative.Set1,height=500,orientation="v")

fig.show()
import plotly.graph_objects as go

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']



fig = go.Figure(data=[go.Pie(labels=who_group["WHO Region"],

                             values=who_group['No. of cases'])])

fig.update_traces(hoverinfo='label+percent', textinfo='label+percent',textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(width=800,

    height=600)

fig.show()
pio.templates.default = "plotly_dark"

fig = px.bar(who_group.sort_values("No. of deaths",ascending=False)[::-1],y="No. of deaths",x ="WHO Region",text="No. of deaths",

             title="WHO regions with highest number of Deaths from 2000 to 2018",

             color_discrete_sequence= px.colors.qualitative.Set1,height=500,orientation="v")

fig.show()
import plotly.graph_objects as go

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']



fig = go.Figure(data=[go.Pie(labels=who_group["WHO Region"],

                                 values=who_group['No. of deaths'])])

fig.update_traces(hoverinfo='label+percent', textinfo='label+percent',textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(width=800,

    height=600)

fig.show()
def plot_treemap_cases(dataframe,variable,value):

    fig = px.treemap(dataframe.sort_values(by=value,ascending=False).reset_index(drop=True),

                     path=[variable],values=value,title=("Highest number of Cases from 2000 to 2018 according to" + str(variable) + "<br>"),

                     color_discrete_sequence = px.colors.qualitative.Set1)

    fig.data[0].textinfo = 'label+text+value'

    

    fig.show()

    

def plot_treemap_deaths(dataframe,variable,value):

    fig = px.treemap(dataframe.sort_values(by=value,ascending=False).reset_index(drop=True),

                     path=[variable],values=value,title=("Highest number of Death Cases from 2000 to 2018 according to" + str(variable) + "<br>"),

                     color_discrete_sequence = px.colors.qualitative.Set1)

    fig.data[0].textinfo = 'label+text+value'

    

    fig.show()

  
plot_treemap_cases(df_group,"Country","No. of cases")
plot_treemap_deaths(df_group,"Country","No. of deaths")
year_group= df.groupby("Year")[["No. of cases","No. of deaths"]].sum().reset_index()

year_group.head()
#whole = df_n.groupby('Date')['Date','Confirmed','Deaths','Recovered','Active'].sum().reset_index()

fig = make_subplots(rows=1,cols=2,column_titles = ('No. of cases','No. of deaths'))



trace_1 = go.Scatter(x=year_group['Year'],y=year_group['No. of cases'],name='Cases',opacity=0.9,mode='lines+markers',line_color='blue')



trace_2 = go.Scatter(x=year_group['Year'],y=year_group['No. of deaths'],name='Deaths',opacity=0.9,mode='lines+markers',line_color='red')



fig.append_trace(trace_1,1,1)

fig.append_trace(trace_2,1,2)



fig.update_layout(title_text="Spread of Malaria according to Year")

fig.show()
fig = px.choropleth(df_group,locationmode="country names",locations ="Country",hover_data = ["Country","No. of cases","No. of deaths"],

                    hover_name = "Country",color="Country",title="Situation of Malaria Over the World"

)

fig.show()
bangladesh = df[df["Country"] == "Bangladesh"]

bangladesh.tail()
#whole = df_n.groupby('Date')['Date','Confirmed','Deaths','Recovered','Active'].sum().reset_index()

fig = make_subplots(rows=1,cols=2,column_titles = ('No. of cases','No. of deaths'))



trace_1 = go.Scatter(x=bangladesh['Year'],y=bangladesh['No. of cases'],name='Cases',opacity=0.9,mode='lines+markers',line_color='blue')



trace_2 = go.Scatter(x=bangladesh['Year'],y=bangladesh['No. of deaths'],name='Deaths',opacity=0.9,mode='lines+markers',line_color='red')



fig.append_trace(trace_1,1,1)

fig.append_trace(trace_2,1,2)



fig.update_layout(title_text="Malaria Report of Bangladesh")

fig.show()