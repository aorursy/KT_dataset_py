import json

import os



import folium

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns
!ls ../input/hospitals-count-in-india-statewise
df = pd.read_csv("../input/hospitals-count-in-india-statewise/Hospitals count in India - Statewise.csv")

df = df.rename(columns={'Number of hospitals in public sector': 'Number_of_hospitals_in_public_sector',

                        'Number of hospitals in private sector': 'Number_of_hospitals_in_private_sector',

                        'Total number of hospitals (public+private)': 'Total_number_of_hospitals_public_private'},)
df
df.info()
feature_columns = ['Number_of_hospitals_in_public_sector', 'Number_of_hospitals_in_private_sector', 'Total_number_of_hospitals_public_private']
def convert_str_with_commma2int(x):

    if type(x) is str:

        return int(x.replace(',',''))

    else:

        return x



for col in feature_columns:

    df[col] = df[col].map(convert_str_with_commma2int)

    

# I recalculate total number because total number of Chhattisgarh seems wrong.

df["Total_number_of_hospitals_public_private"] = df["Number_of_hospitals_in_public_sector"] + df["Number_of_hospitals_in_private_sector"]

    

df.head()
df.describe()
g_public_h_hist = sns.distplot(df["Number_of_hospitals_in_public_sector"], kde=False, rug=False)

g_public_h_hist.set_title("Number_of_hospitals_in_public_sector")
g_private_h_hist = sns.distplot(df["Number_of_hospitals_in_private_sector"], kde=False, rug=False)

g_private_h_hist.set_title("Number_of_hospitals_in_private_sector")
g_total_h_hist = sns.distplot(df["Total_number_of_hospitals_public_private"], kde=False, rug=False)

g_total_h_hist.set_title("Total_number_of_hospitals_public_private")
df_sumup = df.groupby('States/UTs', as_index=False).sum()

df_sumup.head()
fig = px.pie(df_sumup, values='Number_of_hospitals_in_public_sector', names='States/UTs', title='Number_of_hospitals_in_public_sector')

fig.show()
fig = px.pie(df_sumup, values='Number_of_hospitals_in_private_sector', names='States/UTs', title='Number_of_hospitals_in_public_sector')

fig.show()
fig = px.pie(df_sumup, values='Total_number_of_hospitals_public_private', names='States/UTs', title='Number_of_hospitals_in_public_sector')

fig.show()
plt.figure(figsize=(20, 10))

g = sns.scatterplot(data=df, x="Number_of_hospitals_in_public_sector", y="Number_of_hospitals_in_private_sector")

g.set_title("Relation between number of hospitals in public sector and private sector")
plt.figure(figsize=(20, 10))

g = sns.regplot(data=df, x="Number_of_hospitals_in_public_sector", y="Number_of_hospitals_in_private_sector", ci=95)

g.set_title("Relation between number of hospitals in public sector and private sector with lenear regression and 95% confidence interval")
df_high_public = df[(df["Number_of_hospitals_in_public_sector"] <= 2000) &

                    ( 700 <= df["Number_of_hospitals_in_public_sector"] ) & 

                    (df["Number_of_hospitals_in_private_sector"] <= 1000)]
df_high_public
df_high_private = df[(df["Number_of_hospitals_in_public_sector"] <= 1000) &

                     ( 1800 <= df["Number_of_hospitals_in_private_sector"] ) & 

                     (df["Number_of_hospitals_in_private_sector"] <= 4000) | 

                     (df["Number_of_hospitals_in_private_sector"] > 6000)]
df_high_private
#I refered https://www.kaggle.com/niharika41298/covid-19-india-analysis



with open('../input/indian-state-geojson-data/india_state_geo.json') as file:

    geojsonData = json.load(file)

        

for i in geojsonData['features']:

    i['id'] = i['properties']['NAME_1']
#To display area, rename "Himachal Pradesh 8" and "Orissa".

df_high_public["States/UTs"][19] = 'Himachal Pradesh'

df_high_public["States/UTs"][28] = 'Orissa'
map_choropleth_high_public = folium.Map(location = [20.5937,78.9629], zoom_start = 4)
folium.Choropleth(geo_data=geojsonData,

                 data=df_high_public,

                 name='CHOROPLETH',

                 key_on='feature.id',

                 columns = ['States/UTs','Number_of_hospitals_in_public_sector'],

                 fill_color='YlOrRd',

                 fill_opacity=0.7,

                 line_opacity=0.8,

                 legend_name='High rate in public sector hospital',

                 highlight=True).add_to(map_choropleth_high_public)



folium.LayerControl().add_to(map_choropleth_high_public)

display(map_choropleth_high_public)
map_choropleth_high_private = folium.Map(location = [20.5937,78.9629], zoom_start = 4)
folium.Choropleth(geo_data=geojsonData,

                 data=df_high_private,

                 name='CHOROPLETH',

                 key_on='feature.id',

                 columns = ['States/UTs','Number_of_hospitals_in_private_sector'],

                 fill_color='YlOrRd',

                 fill_opacity=0.7,

                 line_opacity=0.8,

                 legend_name='High rate in private sector hospital',

                 highlight=True).add_to(map_choropleth_high_private)



folium.LayerControl().add_to(map_choropleth_high_private)

display(map_choropleth_high_private)