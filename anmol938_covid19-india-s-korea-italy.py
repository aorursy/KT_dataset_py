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



# Any results you write to the current directory are saved as output.
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/COVID19_line_list_data.csv")

covid19_italy_province = pd.read_csv("../input/covid19_italy_province.csv")

covid19_italy_region = pd.read_csv("../input/covid19_italy_region.csv")

covid_19_data = pd.read_csv("../input/covid_19_data.csv")

covid_19_india = pd.read_csv("../input/covid_19_india.csv")

population_india_census2011 = pd.read_csv("../input/population_india_census2011.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/time_series_covid_19_recovered.csv")
#Credit Goes to Parul Pandey 

#https://www.kaggle.com/parulpandey

#Have used her notebook to practice and learn made some normal changes for better understanding

#

import pandas as pd

df = pd.read_excel('../input/Covid cases in India.xlsx')

df.head()

df_India = df.copy()

df_India.head()
#Reading the Indian Coordinates 

India_coord = pd.read_excel('../input/Indian Coordinates.xlsx')

India_coord.head()
#day by day cases in countries like India Korea and Italy && and than there comparison



dbd_India = pd.read_excel('../input/per_day_cases.xlsx', sheet_name = 'India')

dbd_Italy = pd.read_excel('../input/per_day_cases.xlsx', sheet_name = 'Italy')

dbd_Korea = pd.read_excel('../input/per_day_cases.xlsx', sheet_name = 'Korea')

dbd_all   = pd.read_excel('../input/per_day_cases.xlsx', sheet_name = 'Comparison')
dbd_India.head()

dbd_Korea.head()

dbd_Italy.head()

dbd_all.head()
df.columns
# lets check the total cases in India

print(df.head())

df.drop(['S. No.'], axis=1, inplace = True)

print("\n------",df.head())
#Lets add another column into this dataframe that calculates the total number of covid19 cases in India

df['Total_Cases'] = df['Total Confirmed cases (Indian National)'] + df['Total Confirmed cases ( Foreign National )']
df.head()
total_cases = df.Total_Cases.sum()

print("Total Number of covid19 cases in India including foreign as well as Indians---> ",total_cases)
#higlighting the maximum in recovered , Deaths, Total_cases

def highlight_max(s):

    result = []

    is_max = s == s.max()

    for v in is_max:

        if v:

            result.append('background-color:pink')

        else:

            result.append('')

    return result        

        



df.style.apply(highlight_max, subset = ['Recovered','Deaths','Total_Cases'])
#Practice or Checking  of somthing in between

#-------------------------------------------------------------------------------------------------------------

# list_mer = [df.Recovered, df.Deaths, df.Total_Cases]                                                     

# list_mer[0]

# #print(list_mer[0],"\n------------------",list_mer[1],"\n--------------",list_mer[2])

# type(list_mer)

# #is_max = list_mer == list_mer.max()

# for ele in list_mer:

#     print(ele)

#-------------------------------------------------------------------------------------------------------------

df.columns
#based on states & Union terrotories for covid 19

x = df.groupby('Name of State / UT')['Total_Cases'].sum().sort_values(ascending = False).to_frame()

x.style.background_gradient(cmap='Reds')
import plotly_express as px

#another type of visual could be a bar type

fig = px.bar(df.sort_values('Total_Cases',ascending = True), x = 'Total_Cases', y = 'Name of State / UT', orientation ='h', width=700, height=700, range_x=[0, max(df['Total_Cases'])])

fig.show()
#Nationals vs Foreign Cases in states & union territories

import plotly.graph_objects as go 

from plotly.subplots import make_subplots



fig = make_subplots(rows=1, cols=2, subplot_titles=("National Cases","Foreign Cases"))

temp = df.sort_values('Total Confirmed cases (Indian National)',ascending=False)

# type(temp)

# temp.head()

fig.add_trace(go.Bar(y = temp['Total Confirmed cases (Indian National)'], x = temp['Name of State / UT'], marker=dict(color=temp['Total Confirmed cases (Indian National)'], coloraxis="coloraxis")),1,1  )



temp1 = df.sort_values('Total Confirmed cases ( Foreign National )',ascending=False)



fig.add_trace(go.Bar(x = temp1['Name of State / UT'], y = temp1['Total Confirmed cases ( Foreign National )'], marker = dict(color=temp['Total Confirmed cases ( Foreign National )'], coloraxis="coloraxis")),1,2 )



fig.show()

India_coord.head()

df.head()
#THIS is possible that this map wont be visible in your IDE's or jupyter Notebook, but on kaggle

# It will surely work

import folium

#lets merge the India_coord and df dataframes on Name of State/UT

df_full = pd.merge(India_coord,df,on='Name of State / UT')

map = folium.Map(location=[20, 80], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Total_Cases'], df_full['Name of State / UT']):

    folium.CircleMarker([lat, lon],radius=value*0.8,popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)

map



#Confirmed vs Recovered Figures

import matplotlib.pyplot as plt

import seaborn as sns



f, ax = plt.subplots(figsize=(12, 8))

data = df_full[['Name of State / UT','Total_Cases','Recovered','Deaths']]

data.sort_values('Total_Cases',ascending=False,inplace=True)

sns.set_color_codes("pastel")

sns.barplot(x="Total_Cases", y="Name of State / UT", data=data,

            label="Total", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Name of State / UT", data=data,

            label="Recovered", color="g")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35), ylabel="",

       xlabel="Cases")

#Rise in covid 19 cases in India

print(dbd_India.dtypes)# date time is already converted into datetime type  so no problems

fig = go.Figure()

fig.add_trace(go.Scatter(x=dbd_India['Date'], y = dbd_India['Total Cases'], mode='lines+markers', name = 'Total Cases') )

fig.update_layout(title= "Trend of Covid 19 in India",xaxis_title="Time",yaxis_title="Cumulative Covid-19 Cases")





fig.show()
#ploting the new caes in india

dbd_India.head()

import plotly_express as px

fig = px.bar(dbd_India, x = 'Date', y = 'New Cases' )

fig.update_layout(title= 'New cases of Covid 19 per Day',xaxis_title='Time', yaxis_title='New Cases per day',height=400)

fig.show()
#Confirmed Cases in India

fig = px.bar(dbd_Italy, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=500,width=700,

             title='Confirmed Cases in India', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()

print("\n")

fig = px.bar(dbd_Korea, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=500,width=700,

             title='Confirmed Cases in Korea', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()

print("\n")



fig = px.bar(dbd_Korea, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=500,width=700,

             title='Confirmed Cases in South Korea', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(50, 42, 42)')

fig.show()

print("\n")



# Comaprison between S korea Italy India

import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("S.Korea","Italy", "India"))



fig.add_trace(go.Bar(x=dbd_Korea['Date'], y=dbd_Korea['Total Cases'],

                    marker=dict(color=dbd_Korea['Total Cases'], coloraxis="coloraxis")),

              1, 1)



fig.add_trace(go.Bar(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'],

                    marker=dict(color=dbd_Italy['Total Cases'], coloraxis="coloraxis")),

              1, 2)



fig.add_trace(go.Bar(x=dbd_India['Date'], y=dbd_India['Total Cases'],

                    marker=dict(color=dbd_India['Total Cases'], coloraxis="coloraxis")),

              2, 1)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Total Confirmed cases(Cumulative)")



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
# Trend of covid 19 cases in Italy, S.Korea, India

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,

    specs=[[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("S.Korea","Italy", "India"))



fig.add_trace(go.Scatter(x=dbd_Korea['Date'], y=dbd_Korea['Total Cases'],

                    marker=dict(color=dbd_Korea['Total Cases'], coloraxis="coloraxis")),

              1, 1)



fig.add_trace(go.Scatter(x=dbd_Italy['Date'], y=dbd_Italy['Total Cases'],

                    marker=dict(color=dbd_Italy['Total Cases'], coloraxis="coloraxis")),

              1, 2)



fig.add_trace(go.Scatter(x=dbd_India['Date'], y=dbd_India['Total Cases'],

                    marker=dict(color=dbd_India['Total Cases'], coloraxis="coloraxis")),

              2, 1)



fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False,title_text="Trend of Coronavirus cases")



fig.update_layout(plot_bgcolor='rgb(0, 0, 0)')

fig.show()
#Trend after passing the 100 cases

import plotly.graph_objects as go

import numpy as np



title = 'Main Source for News'

labels = ['S.Korea', 'Italy', 'India']

colors = ['rgb(0,150,0)', 'rgb(150,0,0)', 'rgb(4,197,198)']



mode_size = [8, 8, 12]

line_size = [2, 2, 6]



fig = go.Figure()





fig.add_trace(go.Scatter(x=dbd_Korea['Days after surpassing 100 cases'], 

                 y=dbd_Korea['Total Cases'],mode='lines',

                 name=labels[0],

                 line=dict(color=colors[0], width=line_size[0]),            

                 connectgaps=True,

    ))

fig.add_trace(go.Scatter(x=dbd_Italy['Days after surpassing 100 cases'], 

                 y=dbd_Italy['Total Cases'],mode='lines',

                 name=labels[1],

                 line=dict(color=colors[1], width=line_size[1]),            

                 connectgaps=True,

    ))



fig.add_trace(go.Scatter(x=dbd_India['Days after surpassing 100 cases'], 

                 y=dbd_India['Total Cases'],mode='lines',

                 name=labels[2],

                 line=dict(color=colors[2], width=line_size[2]),            

                 connectgaps=True,

    ))

    

    

    

annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Days after surpassing 100 cases ',

                              font=dict(family='Arial',

                                        size=12,

                                        color='rgb(190,190,190)'),

                              showarrow=False))



fig.update_layout(annotations=annotations,plot_bgcolor='white',yaxis_title='Cumulative cases')



fig.show()