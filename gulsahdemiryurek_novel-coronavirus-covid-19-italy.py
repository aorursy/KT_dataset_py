# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
dprovince=pd.read_csv("/kaggle/input/italy-covid19/covid19-ita-province.csv")

dnational=pd.read_csv("/kaggle/input/italy-covid19/covid-nationality.csv")

dregions=pd.read_csv("/kaggle/input/italy-covid19/covid19-ita-regions.csv")
d2=dnational.drop(["Unnamed: 0","state"],axis=1)

import plotly.graph_objects as go

ab=d2.tail(1).values.T

fig = go.Figure(data=[go.Table(

    header=dict(values=d2.tail(1).columns,

                line_color='darkslategray',

                fill_color='lightskyblue',

                align='left',font_size=10),

    cells=dict(values=list(ab), # 2nd column

               line_color='darkslategray',

               fill_color='lightcyan',

               align='left',font_size=12,height=40))

])



fig.update_layout(width=1600, height=250,title="Last Status in Italy")

fig.show()

d3=d2.drop(["date","new_confirmed_cases","note_it","note_en"],axis=1)

c=[]

for i in range(38):

    x=d3.iloc[i+1]-d3.iloc[i]

    c.append(x)
ax=pd.DataFrame(c)

ax.home_quarantine=d3.home_quarantine

da=pd.DataFrame(d3.loc[0])

dx=da.T 

ax1= dx.append(ax, ignore_index=True)

ax1["percent_test_confirmed"]=(ax1.total_confirmed_cases/ax1.swabs_made)*100

ax1["date"]=d2.date

ax1.set_index("date",inplace=True)

               
ax1.info()
def highlight_max(s):    

    is_max = s == s.max()

    return ['background-color: salmon' if v else "background-color: mistyrose" for v in is_max]

ax1.style.apply(highlight_max)
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

                x=dnational.date,

                y=dnational['hospitalized_with_symptoms'],

                name="Hospitalized with Symptoms",

                mode="markers",marker=dict(size=12,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=dnational.date,

                y=dnational['home_quarantine'],

                name="Home Quarantine",

                mode="markers",marker=dict(size=12,symbol=22,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=dnational.date,

                y=dnational["intensive_care"],

                name="Intensive Care",

                mode="markers",marker=dict(size=12,symbol=20, line=dict(width=2,

                                        color='DarkSlateGrey')),

                opacity=0.8))



# Use date string to set xaxis range

fig.update_layout(title_text="Hospitalized with Symptoms,Home Quarantine and Intensive Care Cases",plot_bgcolor='azure',width=1000)

fig.update_xaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Date")

fig.update_yaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Number")



fig.show()
from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=3, cols=1)



fig.append_trace(go.Bar(

    x=dnational.date,

    y=dnational.total_confirmed_cases,

    name="Total Confirmed Cases",

    text=dnational.total_confirmed_cases,

    textposition="outside",

), row=1, col=1)



fig.append_trace(go.Bar(

    x=dnational.date,

    y=dnational.recovered,

    name="Recovered Cases",

    text=dnational.recovered,

    textposition="outside",

), row=2, col=1)



fig.append_trace(go.Bar(

    x=dnational.date,

    y=dnational.deaths,

    name="Deaths",

    text=dnational.deaths,

    textposition="outside",

), row=3, col=1)





fig.update_layout(height=1000, width=1000, title_text="Total Confirmed , Recovered and Death Cases ")

fig.update_xaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Date")

fig.update_yaxes(showline=True, linewidth=2, linecolor='dimgray', mirror=True,gridcolor='silver',title="Number")

fig.show()
laststatus=dregions[dregions.date=="2020-03-26T17:00:00"]

import plotly.express as px



fig = px.scatter_mapbox(laststatus, lat="lat", lon="long", hover_name="region", hover_data=["total_confirmed_cases", "deaths","recovered","swabs_made"],size="total_cases",

                        color_discrete_sequence=["black"], zoom=4, height=400)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
import plotly.express as px



fig = px.treemap(dregions, path=[ 'region','date'],

                  color='total_confirmed_cases', 

                 hover_data=["region",'date',"hospitalized_with_symptoms","intensive_care","total_hospitalized",

                                                            "home_quarantine","recovered","deaths","swabs_made"],

                  color_continuous_scale="magma"

                  )

fig.update_layout(uniformtext=dict(minsize=20),height=1000,width=1000,title="Daily Cases According to Regions")



fig.show()
lombardia=dprovince[dprovince.region=="Lombardia"]

value={"2020-02-24T18:00:00":"24-02","2020-02-25T18:00:00":"25-02","2020-02-26T18:00:00":"26-02","2020-02-27T18:00:00":"27-02","2020-02-28T18:00:00":"28-02",

       "2020-02-29T17:00:00":"29-02","2020-03-01T17:00:00":"01-03","2020-03-02T17:00:00":"02-03","2020-03-03T17:00:00":"03-03","2020-03-04T17:00:00":"04-03",

      "2020-03-05T17:00:00":"05-03","2020-03-06T18:00:00":"06-03","2020-03-07T18:00:00":"07-03","2020-03-08T18:00:00":"08-03","2020-03-09T18:00:00":"09-03",

      "2020-03-10T18:00:00":"10-03","2020-03-11T17:00:00":"11-03","2020-03-12T17:00:00":"12-03","2020-03-13T17:00:00":"13-03","2020-03-14T17:00:00":"14-03",

      "2020-03-15T17:00:00":"15-03","2020-03-16T17:00:00":"16-03","2020-03-17T17:00:00":"17-03","2020-03-18T17:00:00":"18-03","2020-03-19T17:00:00":"19-03",

      "2020-03-20T17:00:00":"20-03","2020-03-21T17:00:00":"21-03","2020-03-22T17:00:00":"22-03","2020-03-23T17:00:00":"23-03","2020-03-24T17:00:00":"24-03",

      "2020-03-25T17:00:00":"25-03","2020-03-26T17:00:00":"26-03", "2020-03-27T17:00:00":"27-03","2020-03-28T17:00:00":"28-03"}

lombardia.date=lombardia.date.replace(value)
import seaborn as sns 

import matplotlib.pyplot as plt

#plt.Figure(figsize=(80,50))

plt.rcParams["axes.labelsize"] = 28

g = sns.catplot(x="date", y="total_cases",

                 col="province",col_wrap=5,palette="bone_r",

                data=lombardia , kind="point",

                 aspect=.9)

g.set_xticklabels(rotation=90)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Total Cases According to Province in Lombardia',fontsize=25)

plt.show()
