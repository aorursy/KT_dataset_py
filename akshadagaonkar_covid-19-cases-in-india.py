import pandas as pd

import pandas as pd 

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import plotly.express as px



!pip install pywaffle

from pywaffle import Waffle



import plotly.graph_objects as go

from plotly.subplots import make_subplots



import folium 

from folium import plugins



import warnings

warnings.filterwarnings('ignore')

data= pd.read_csv("../input/coronavirus-cases-in-india/Covid cases in India.csv")



India_coord = pd.read_csv('../input/coronavirus-cases-in-india/Indian Coordinates.csv')



dbd=pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx')

dbd_India = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')

dbd_Italy = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Italy")

dbd_Korea = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Korea")

data.head()

data.drop(["S. No."], axis=1, inplace=True)
data['Total cases'] = data['Total Confirmed cases (Indian National)'] + data['Total Confirmed cases ( Foreign National )'] 

data['Active cases'] = data['Total cases'] - (data['Cured/Discharged/Migrated'] + data['Deaths'])

print('Total number of Confirmed cases across India:', data['Total cases'].sum())

print('Total number of Active cases across India:', data['Active cases'].sum())

print('Total number of Cured/Discharged/Migrated cases across India:', data['Cured/Discharged/Migrated'].sum())

print('Total number of Deaths due to COVID 2019 across India:', data['Deaths'].sum())

print('Total number of States/UTs affected:', len(data['Name of State / UT']))

data
def highlight_max(s):

    is_max = s == s.max()

    return['background-color : lightcoral' if v else '' for v in is_max]



#data.style.apply(highlight_max, subset= pd.IndexSlice[:,["Total Confirmed cases (Indian National)", "Total Confirmed cases ( Foreign National )"]])

data.style.apply(highlight_max, subset= pd.IndexSlice[:,["Cured/Discharged/Migrated", "Deaths", "Total cases", "Active cases"]])

s_ut = data.groupby('Name of State / UT')['Active cases'].sum().sort_values(ascending=False).to_frame()

s_ut.style.background_gradient(cmap= 'Reds')
fig = px.bar(data.sort_values('Active cases', ascending=False).sort_values('Active cases', ascending=True), 

             x="Active cases", y="Name of State / UT", title='Total Active Cases', text='Active cases', orientation='h', 

             width=1000, height=700, range_x = [0, max(data['Active cases'])])

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.show()
totals=[data['Active cases'].sum(),data['Cured/Discharged/Migrated'].sum() ,data['Deaths'].sum()]

labels='Active cases','Recovered','Death'

explode= (0.15,0.15,0)



colors = ['yellowgreen','lightskyblue', 'lightcoral']

plt.pie(totals, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.tight_layout()

plt.show()
fig = make_subplots(rows=1, cols=2, subplot_titles=("National Cases","Foreign Cases"))



temp = data.sort_values('Total Confirmed cases (Indian National)', ascending=False).sort_values('Total Confirmed cases (Indian National)', ascending=False)



fig.add_trace(go.Bar( y=temp['Total Confirmed cases (Indian National)'], x=temp["Name of State / UT"],  

                     marker=dict(color=temp['Total Confirmed cases (Indian National)'], coloraxis="coloraxis")),

              1, 1)

                     

temp1 = data.sort_values('Total Confirmed cases ( Foreign National )', ascending=False).sort_values('Total Confirmed cases ( Foreign National )', ascending=False)



fig.add_trace(go.Bar( y=temp1['Total Confirmed cases ( Foreign National )'], x=temp1["Name of State / UT"],  

                     marker=dict(color=temp1['Total Confirmed cases ( Foreign National )'], coloraxis="coloraxis")),

              1, 2)                     

                     



fig.update_layout(coloraxis=dict(colorscale='rdbu'), showlegend=False,title_text="National vs Foreign Cases",plot_bgcolor='rgb(250, 242, 242)')

fig.show()



full = pd.merge(India_coord, data, on='Name of State / UT')

map = folium.Map(location=[20, 80], zoom_start=3.5, tiles='Stamen Toner')



for lat, lon, value, name in zip(full['Latitude'], full['Longitude'], full['Active cases'], full['Name of State / UT']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.7,

                        popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.3 ).add_to(map)

map
fig = go.Figure()

fig.add_trace(go.Scatter(x=dbd_India['Date'], y=dbd_India['Total Cases'], mode='lines+markers',name='Total Cases'))

fig.update_layout(title_text='Trend of Coronavirus Cases in India',plot_bgcolor='rgb(250, 242, 242)')



fig.show()
fig = px.bar(dbd_India, x="Date", y="New Cases", barmode='group', height=400)

fig.update_layout(title_text='New Coronavirus Cases in India per day',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
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