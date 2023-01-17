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
from pandas.io.json import json_normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh


import json
import urllib

import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px

import folium

import warnings
warnings.filterwarnings("ignore")


# !pip install folium==0.10.1
cases_by_date=json.load(urllib.request.urlopen("https://api.covid19india.org/data.json"))     # done
state_district_wise=json.load(urllib.request.urlopen("https://api.covid19india.org/v2/state_district_wise.json"))
travel_history=json.load(urllib.request.urlopen("https://api.covid19india.org/travel_history.json"))
raw_data=json.load(urllib.request.urlopen("https://api.covid19india.org/raw_data.json"))
state_wise=json.load(urllib.request.urlopen("https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise")) #done


raw_data = json_normalize(raw_data["raw_data"]) 
cases_by_date=json_normalize(cases_by_date["cases_time_series"]) 
travel_history=json_normalize(travel_history["travel_history"]) 
state_wise=json_normalize(state_wise["data"]["statewise"])   
state_district_wise=json_normalize(state_district_wise)  
raw_data=pd.get_dummies(raw_data, columns=["currentstatus"])
state_wise["Percent"]=np.nan
for i in range (0,len(state_wise["Percent"]) ):
    state_wise["Percent"][i]=round(((state_wise["confirmed"][i]/state_wise["confirmed"].sum())*100),2)
  
cor=pd.read_json('{"state":{"0":"Kerala","1":"Delhi","2":"Telengana","3":"Rajasthan","4":"Haryana","5":"Uttar Pradesh","6":"Tamil Nadu","7":"Union Territory of Ladakh","8":"Karnataka","9":"Maharashtra","10":"Punjab","11":"Union Territory of Jammu and Kashmir","12":"Andhra Pradesh","13":"Uttarakhand","14":"Odisha","15":"Puducherry","16":"West Bengal","17":"Chhattisgarh","18":"Union Territory of Chandigarh","19":"Gujarat","20":"Chandigarh","21":"Himachal Pradesh","22":"Jammu and Kashmir","23":"Ladakh","24":"Madhya Pradesh","25":"Bihar","26":"Manipur","27":"Mizoram","28":"Goa","29":"Andaman and Nicobar Islands","30":"Arunachal Pradesh","31":"Assam","32":"Dadra and Nagar Haveli","33":"Daman and Diu","34":"Jharkhand","35":"Lakshadweep","36":"Meghalaya","37":"Sikkim","38":"Telangana","39":"Nagaland","40":"Tripura", "41":"Dadra and Nagar Haveli and Daman and Diu" },"Latitude":{"0":10.8505,"1":28.7041,"2":18.1124,"3":27.0238,"4":29.0588,"5":26.8467,"6":11.1271,"7":34.2996,"8":15.3173,"9":19.7515,"10":31.1471,"11":33.7782,"12":15.9129,"13":30.0668,"14":20.9517,"15":11.9416,"16":22.9868,"17":21.2787,"18":30.7333,"19":22.2587,"20":30.7333,"21":31.1048,"22":33.7782,"23":34.2996,"24":22.9734,"25":25.0961,"26":24.6637,"27":23.1645,"28":15.2993,"29":11.7401,"30":28.218,"31":31.1048,"32":20.1809,"33":20.4283,"34":23.6102,"35":10.0,"36":25.467,"37":27.533,"38":18.1124,"39":26.1584,"40":23.9408, "41":20.1809},"Longitude":{"0":76.2711,"1":77.1025,"2":79.0193,"3":74.2179,"4":76.0856,"5":80.9462,"6":78.6569,"7":78.2932,"8":75.7139,"9":75.7139,"10":75.3412,"11":76.5762,"12":79.74,"13":79.0193,"14":85.0985,"15":79.8083,"16":87.855,"17":81.8661,"18":76.7794,"19":71.1924,"20":76.7794,"21":77.1734,"22":76.5762,"23":78.2932,"24":78.6569,"25":85.3131,"26":93.9063,"27":92.9376,"28":74.124,"29":92.6586,"30":94.7278,"31":77.1734,"32":73.0169,"33":72.8397,"34":85.2799,"35":73.0,"36":91.3662,"37":88.5122,"38":79.0193,"39":94.5624,"40":91.9882,"41":73.0169}}')
cor.to_csv("India Coordinates.csv")
state_wise=pd.merge(state_wise, cor, how='left', on='state')
state_wise.columns
df1_sum=state_wise
temp=df1_sum.index[df1_sum['state']=='State Unassigned'][0]
df1_sum.drop([temp], axis=0, inplace=True)
# For entitre India
import folium
India = folium.Map(location=[20.5937,78.9629], tiles='cartodbpositron', min_zoom=4, max_zoom=8, zoom_start=4.4)

for i in range(0, len(df1_sum)):
    folium.CircleMarker(
        location=[df1_sum.iloc[i]['Latitude'],df1_sum.iloc[i]['Longitude']],
                  color='crimson',
                  tooltip = '<li><bold>State/UnionTerritory : '+str(df1_sum.iloc[i]['state'])+
                            
                            '<li><bold>Total Count : '+str(df1_sum.iloc[i]['confirmed'])+
                            '<li><bold>Recovered : '+str(df1_sum.iloc[i]['recovered'])+
                            '<li><bold>Active : '+str(df1_sum.iloc[i]['active'])+
                            '<li><bold>Died : '+str(df1_sum.iloc[i]['deaths'])+
                            '<li><bold>Percentage WRT India : '+str(df1_sum.iloc[i]['Percent']),
                                radius=5).add_to(India)
    
India

cases_by_date10=cases_by_date.tail(15)


fig = go.Figure(data=[
    go.Bar(name='Total', x=cases_by_date10.date, y=cases_by_date10.totalconfirmed , text=cases_by_date10.totalconfirmed),
    go.Bar(name='Recovered', x=cases_by_date10.date, y=cases_by_date10.totalrecovered, text=cases_by_date10.totalrecovered),
    go.Bar(name='Died', x=cases_by_date10.date, y=cases_by_date10.totaldeceased, text=cases_by_date10.totaldeceased)
])


fig.update_traces(textposition='outside')

fig.update_layout(barmode='group', title="Bar Chart for last 15 Days", paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'   )
fig.show()
# Create traces
fig = go.Figure()

fig.add_trace(go.Scatter(name='Total', x=cases_by_date10.date, y=cases_by_date10.totalconfirmed,
                    mode='lines+markers'
                    ))
fig.add_trace(go.Scatter(name='Died', x=cases_by_date10.date, y=cases_by_date10.totaldeceased,
                     mode='lines+markers',
                    ))
fig.add_trace(go.Scatter(name='Recovered', x=cases_by_date10.date, y=cases_by_date10.totalrecovered,
                    mode='lines+markers',
                    ))

fig.update_layout(title="Line Chart for last 15 Days", paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)'   )

fig.show()