%matplotlib inline

import requests

import pandas as pd

import numpy as np

import json

import matplotlib.pyplot as plt

import folium

odisha = folium.Map(location=[20.9517,85.098], zoom_start=4,max_zoom=6,min_zoom=4,height=500,width="80%")

# for i in range(0,len(df_india[df_india['confirmed']>0].index)):

#     folium.Circle(

#         location=[df_india.iloc[i]['Lat'], df_india.iloc[i]['Long']],

#         tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_india.iloc[i].name+"</h5>"+

#                     "<hr style='margin:15px;'>"+

#                     "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

#         "<li>Confirmed: "+str(df_india.iloc[i]['confirmed'])+"</li>"+

#         "<li>Active:   "+str(df_india.iloc[i]['active'])+"</li>"+

#         "<li>Recovered:   "+str(df_india.iloc[i]['recovered'])+"</li>"+

#         "<li>Deaths:   "+str(df_india.iloc[i]['deaths'])+"</li>"+

#         "</ul>"

#         ,

#         radius=(int(np.log2(df_india.iloc[i]['confirmed']+1)))*15000,

#         color='	#0000FF',

#         fill_color='red',

#         fill_opacity=0.9,

#         fill=True).add_to(india)

odisha
odisha_data_json = requests.get('https://api.covid19india.org/state_district_wise.json').json()

df_odisha = pd.io.json.json_normalize(odisha_data_json['Odisha'])

#df_odisha = df_odisha.set_index("Odisha")

df_odisha.head(4)
df_odisha.columns
df_dist = df_odisha.rename(columns = {'districtData.Khordha.confirmed': 'Khordha','districtData.Bhadrak.confirmed':'Bhadrak',

                                      'districtData.Cuttack.confirmed':'Cuttack','districtData.Puri.confirmed':'Puri',

                                     'districtData.Jajapur.confirmed':'Jajapur','districtData.Kalahandi.confirmed':'Kalahandi',

                                     'districtData.Kendrapara.confirmed':'Kendrapara','districtData.Dhenkanal.confirmed':'Dhenkanal',

                                     'districtData.Sundargarh.confirmed':'Sundargarh'})
df_dist= df_dist.drop(['districtData.Khordha.lastupdatedtime',

       'districtData.Khordha.delta.confirmed',

       'districtData.Bhadrak.lastupdatedtime',

       'districtData.Bhadrak.delta.confirmed',

       'districtData.Cuttack.lastupdatedtime',

       'districtData.Cuttack.delta.confirmed',

       'districtData.Puri.lastupdatedtime',

       'districtData.Puri.delta.confirmed',

       'districtData.Jajapur.lastupdatedtime',

       'districtData.Jajapur.delta.confirmed',

       'districtData.Kalahandi.lastupdatedtime',

       'districtData.Kalahandi.delta.confirmed',

       'districtData.Kendrapara.lastupdatedtime',

       'districtData.Kendrapara.delta.confirmed',

       'districtData.Dhenkanal.lastupdatedtime',

       'districtData.Dhenkanal.delta.confirmed',

       'districtData.Sundargarh.lastupdatedtime',

       'districtData.Sundargarh.delta.confirmed'],axis =1)
df_dist.insert(0, "confirmed", 'Total') 

df_dist = df_dist.set_index("confirmed")

df_dist.style.background_gradient(cmap='Reds',axis=1)

fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

df_dist.plot.barh(ax=ax,title="Confirmed Cases");



#plot.show()
df_dist.transpose().style.background_gradient(cmap='Wistia',axis=1)

# #df_dist.columns
india_data_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()

df_india = pd.io.json.json_normalize(india_data_json['data']['statewise'])

df_india = df_india.set_index("state")
df_india

output = df_india.to_csv('df_india.csv')
total = df_india.sum()

total.name = "Total"

df_t = pd.DataFrame(total,dtype=float).transpose()

df_t["Mortality Rate (per 100)"] = np.round(100*df_t["deaths"]/df_t["confirmed"],2)

df_t.style.background_gradient(cmap='Reds',axis=1)
# <div class="flourish-embed" data-src="story/272414" data-url="https://flo.uri.sh/story/272414/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>
from IPython.core.display import HTML

HTML('''<div class="flourish-embed" data-src="story/272447" data-url="https://flo.uri.sh/story/272447/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')