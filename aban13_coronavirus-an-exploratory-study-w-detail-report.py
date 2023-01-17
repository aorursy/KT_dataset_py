'''Import basic modules.'''

import pandas as pd

import numpy as np





'''Customize visualization

Seaborn and matplotlib visualization.'''

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

import plotly.express as px

import folium



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('aerq4byr7ps',width=600, height=400)
#Reading the dataset

data= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)

data.head()
#Let's look at the various features

data.info()
#Data Cleaning

data['Last Update'] = data['Last Update'].apply(pd.to_datetime)

data['Date'] = data['Date'].apply(pd.to_datetime)

data.drop(['Sno'],axis=1,inplace=True)

data.head()
#Get the data of the latest date from the dataset

from datetime import date

maxDate = max(data['Date'])

df_lastDate = data[data['Date'] >  pd.Timestamp(date(maxDate.year,maxDate.month,maxDate.day))]

df_lastDate.head()
bold('**Present Gobal condition: confirmed, death and recovered**')

print('Globally Confirmed Cases: ',df_lastDate['Confirmed'].sum())

print('Global Deaths: ',df_lastDate['Deaths'].sum())

print('Globally Recovered Cases: ',df_lastDate['Recovered'].sum())
# Countries affected



countries = df_lastDate['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
#Replacing Mainland China with China



df_lastDate['Country'].replace({'Mainland China':'China'},inplace=True)

countries = df_lastDate['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
bold("** COUNTRY WISE CONFIRMED CASES **")

temp = df_lastDate.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()



cm = sns.light_palette("green", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px'),

  ('color', 'black')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(temp.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_table_styles(styles))
def color_red(val):

    """

    Takes a scalar and returns a string with

    the css property `'color: red'` for postive

     cases, black otherwise.

    """

    color = 'red' if val > 0 else 'black'

    return 'color: %s' % color
bold("** Province/State WISE CONFIRMED CASES **")

temp = df_lastDate.groupby(['Country','Province/State']).sum()

(temp.style

  .applymap(color_red,subset=["Deaths"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  )

# Provinces where deaths have taken place

df_lastDate.groupby('Country')['Deaths'].sum().sort_values(ascending=False)[:5]
df_china = df_lastDate[df_lastDate['Country']=='China'][["Province/State","Confirmed","Deaths","Recovered"]]



bold("**Present Scenario of China Condition**")



# Set colormap equal to seaborns light green color palette

cm = sns.light_palette("green", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px'),

  ('color', 'black')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(df_china.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_table_styles(styles))
bars = alt.Chart(df_china.head(10)).mark_bar(color='orange',opacity=0.7).encode(

    x='Confirmed:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['Top 10 Infected State in China'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Confirmed:Q'    

)



(bars + text).properties( height=300, width=600)
temp = df_china[df_china['Recovered']> 0]

bars = alt.Chart(temp.head(10)).mark_bar(color='green',opacity=0.7).encode(

    x='Recovered:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['Top 10 States With Recovered Cases in China'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Recovered:Q'    

)



(bars + text).properties( height=300, width=600)
temp = df_china[df_china['Deaths'] > 0]

bars = alt.Chart(temp.head(10)).mark_bar(color='red',opacity=0.7).encode(

    x='Deaths:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['Top 10 States With Deaths Case in China'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Deaths:Q'    

)



(bars + text).properties( height=300, width=600)
f, ax = plt.subplots(figsize=(12, 8))



sns.barplot(x="Confirmed", y="Province/State", data=df_china,

            label="Confirmed", color="#F0E68C")

sns.barplot(x="Recovered", y="Province/State", data=df_china,

            label="Recovered", color="g")



# Add a legend and informative axis label

ax.set_title('Confirmed vs Recovered figures of Provinces of China', fontsize=20, fontweight='bold', position=(0.53, 1.05))

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=df_china,

            label="Confirmed", color="#F0E68C")



sns.set_color_codes("muted")

sns.barplot(x="Deaths", y="Province/State", data=df_china,

            label="Deaths", color="#DC143C")



# Add a legend and informative axis label

ax.set_title('Confirmed vs Death figures of Provinces of China', fontsize=20, fontweight='bold', position=(0.53, 1.05))

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
# px.choropleth(full_latest_grouped, locations='Country/Region', color='Confirmed',color_continuous_scale="Viridis")



fig = px.choropleth(full_latest_grouped, locations="Country/Region", locationmode='country names', 

                    color="Confirmed", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Sunsetdark", 

                    title='Countries with Confirmed Cases')

fig.update(layout_coloraxis_showscale=False)

fig.show()



fig = px.choropleth(full_latest_grouped[full_latest_grouped['Deaths']>0], locations="Country/Region", locationmode='country names',

                    color="Deaths", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Peach",

                    title='Countries with Deaths Reported')

fig.update(layout_coloraxis_showscale=False)

fig.show()
data['date'] = data['Date'].dt.date

spread = data[data['date'] > pd.Timestamp(date(2020,1,21))]

spread_gl = spread.groupby('date')["Confirmed", "Deaths", "Recovered"].sum().reset_index()

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, subplot_titles=("Confirmed", "Deaths", "Recovered"))



trace1 = go.Scatter(

                x=spread_gl['date'],

                y=spread_gl['Confirmed'],

                name="Confirmed",

                line_color='orange',

                opacity=0.8)

trace2 = go.Scatter(

                x=spread_gl['date'],

                y=spread_gl['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8)



trace3 = go.Scatter(

                x=spread_gl['date'],

                y=spread_gl['Recovered'],

                name="Recovered",

                line_color='green',

                opacity=0.8)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.update_layout(template="ggplot2",title_text = '<b>Global Spread of the Coronavirus Over Time </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
china_data = spread[spread['Country']=='China']

date_con_ch = china_data.groupby('date')['Confirmed','Deaths','Recovered'].sum().reset_index()





from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))



trace1 = go.Scatter(

                x=date_con_ch['date'],

                y=date_con_ch['Confirmed'],

                name="Confirmed",

                line_color='orange',

                opacity=0.8)

trace2 = go.Scatter(

                x=date_con_ch['date'],

                y=date_con_ch['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8)



trace3 = go.Scatter(

                x=date_con_ch['date'],

                y=date_con_ch['Recovered'],

                name="Recovered",

                line_color='green',

                opacity=0.8)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.update_layout(template="ggplot2",title_text = '<b>Spread of the Coronavirus Over Time In China</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
hubei_data = spread[spread['Province/State'] == 'Hubei']



from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))



trace1 = go.Scatter(

                x=hubei_data['date'],

                y=hubei_data['Confirmed'],

                name="Confirmed",

                line_color='orange',

                opacity=0.8)

trace2 = go.Scatter(

                x=hubei_data['date'],

                y=hubei_data['Deaths'],

                name="Deaths",

                line_color='red',

                opacity=0.8)



trace3 = go.Scatter(

                x=hubei_data['date'],

                y=hubei_data['Recovered'],

                name="Recovered",

                line_color='green',

                opacity=0.8)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.update_layout(template="ggplot2",title_text = '<b>Spread of the Coronavirus Over Time In Hubei(Wuhan)</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
df_hubei = sorted_df.loc[sorted_df['PS'] == 'Hubei']

df_lastDate

df_hubei = df_lastDate[df_lastDate['Province/State']=='Hubei'][["Province/State","Confirmed","Deaths","Recovered"]]