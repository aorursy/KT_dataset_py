import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.options.display.max_rows= None

pd.options.display.max_columns= None
df= pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv', parse_dates= True, index_col= 'Sno')

df['Date']= pd.to_datetime(df['Date']).dt.date

df['Last Update']= pd.to_datetime(df['Last Update']).dt.date

df.set_index('Date', inplace= True)

df.head(5)
df.shape
df.describe()
# Checking for NA values

print('Column\t\t#Missing')

df.isna().sum()
df['Country'].replace({'Mainland China': 'China'}, inplace= True)

recent_cp_df= df.groupby(['Country', 'Province/State']).last()

recent_cp_df
recent_cp_df_c= recent_cp_df.groupby('Country').agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})

recent_cp_df_c['Recovery Rate']= recent_cp_df_c['Recovered']/recent_cp_df_c['Confirmed']

recent_cp_df_c['Mortality Rate']= recent_cp_df_c['Deaths']/recent_cp_df_c['Confirmed']

recent_c_df=  df



for i in recent_cp_df_c.index:

    recent_c_df= recent_c_df[(recent_c_df['Country']!=i)]

    

recent_c_df= recent_c_df.groupby(['Country']).last()

recent_c_df.drop(['Province/State', 'Last Update'], axis= 1, inplace= True)

recent_c_df['Recovery Rate']= recent_c_df['Recovered']/recent_c_df['Confirmed']

recent_c_df['Mortality Rate']= recent_c_df['Deaths']/recent_c_df['Confirmed']



recent_df= pd.concat([recent_cp_df_c, recent_c_df], axis= 0)

recent_df
for i in ['Brazil', 'Ivory Coast', 'Mexico']:

    df= df[(df['Country']!=i)]



recent_df_nc= recent_df.drop(['China']).sort_values(['Confirmed'], ascending= False)

recent_df_nc
f, ax = plt.subplots(figsize=(20, 10))

sns.barplot(x= recent_df_nc["Confirmed"], y= recent_df_nc.index, label="Confirmed", color="yellow").set_title('Global Corona outbreak stats for all countries except China', size= 20)

sns.barplot(x= recent_df_nc["Recovered"], y= recent_df_nc.index, label="Recovered", color="green")

sns.barplot(x= recent_df_nc["Deaths"], y= recent_df_nc.index, label="Deaths", color="red")

sns.despine(left= True)

ax.legend(ncol=3, loc="lower right")

ax.set(ylabel="Countries", xlabel="Values")
print('Globally, these are the total numbers reported yet: ')

recent_df.agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 'Recovery Rate': 'mean', 'Mortality Rate': 'mean'}).to_frame()
clist= df['Country'].unique().tolist()



print('Following ' + str(len(clist)) + ' countries were affected: ')

print(clist)
print('Sorted by confirmed cases:')

recent_df.sort_values(['Confirmed'], ascending= False)
print('Sorted by Mortality rate:')

recent_df.sort_values(['Mortality Rate'], ascending= False)
print('Sorted by recovery rate:')

recent_df.sort_values(['Recovery Rate'], ascending= True)
!pip install folium

import folium
coord_df= pd.read_csv('../input/corona-analysis-files/world_coordinates.csv', index_col= 'Country')

coord_df.head()
recent_df= recent_df.join(coord_df, how= 'inner')

recent_df.drop(['Brazil', 'Mexico'], inplace= True)

recent_df
world_map = folium.Map(location=[35.861660, 80.195397], zoom_start= 3, tiles='Stamen Toner')

outbreaks = folium.map.FeatureGroup()



for lt, ln, nm, cnfrm, rec, mor in zip(recent_df['latitude'], recent_df['longitude'], recent_df.index, recent_df['Confirmed'], recent_df['Recovery Rate'], recent_df['Mortality Rate']):

    ss= '<b>Country: </b>' + nm + '<br><b>#Confirmed: </b>' + str(int(cnfrm)) + '<br><b>Recovery rate: </b>' + str(round(rec, 2)) + '<br><b>Mortality rate: </b>' + str(round(mor, 2))

    folium.Marker([lt, ln], popup= ss).add_to(world_map) 

    folium.CircleMarker([lt, ln], radius= 0.05*int(cnfrm), color= 'red').add_to(world_map) 

    

world_map
wc = r'../input/corona-analysis-files/world_countries.json' # geojson file

tscale= np.linspace(0, recent_df['Confirmed'].max()+1, 6, dtype=int).tolist()



world_map = folium.Map(location=[035.861660, 104.195397], zoom_start=2, tiles='Stamen Toner')

world_map.choropleth(

    geo_data= wc,

    data= recent_df,

    columns=[recent_df.index, 'Confirmed'],

    key_on='feature.properties.name',

    threshold_scale= tscale,

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Corona Outbreak strength',

)



world_map
china_recent_p_df= recent_cp_df.loc[['China']].reset_index(level= 0, drop= True)



print('Following ' + str(china_recent_p_df.shape[0]) + ' Chinese provinces were affected: ')

print(china_recent_p_df.index.values)
china_recent_p_df['Recovery Rate']= china_recent_p_df['Recovered']/china_recent_p_df['Confirmed']

china_recent_p_df['Mortality Rate']= china_recent_p_df['Deaths']/china_recent_p_df['Confirmed']



china_recent_p_df_s= china_recent_p_df.sort_values(['Confirmed'], ascending= False)

china_recent_p_df_s
f, ax = plt.subplots(figsize=(20, 10))

sns.set(style="whitegrid")

sns.barplot(x= china_recent_p_df_s["Confirmed"], y= china_recent_p_df_s.index, label="Confirmed", color="yellow").set_title('Corona outbreak stats for all affected provinces in China', size= 20)

sns.barplot(x= china_recent_p_df_s["Recovered"], y= china_recent_p_df_s.index, label="Recovered", color="green")

sns.barplot(x= china_recent_p_df_s["Deaths"], y= china_recent_p_df_s.index, label="Deaths", color="red")

sns.despine(left= True, bottom= True)

ax.legend(ncol=3, loc="lower right")

ax.set(ylabel="Chinese provinces", xlabel="Values")
china_recent_p_df_s.drop('Hubei', inplace= True)



f, ax = plt.subplots(figsize=(20, 10))

sns.barplot(x= china_recent_p_df_s["Confirmed"], y= china_recent_p_df_s.index, label="Confirmed", color="yellow").set_title('Corona outbreak stats for all affected provinces in China except Hubei', size= 20)

sns.barplot(x= china_recent_p_df_s["Recovered"], y= china_recent_p_df_s.index, label="Recovered", color="green")

sns.barplot(x= china_recent_p_df_s["Deaths"], y= china_recent_p_df_s.index, label="Deaths", color="red")

sns.despine(left= True, bottom= True)

ax.legend(ncol=3, loc="lower right")

ax.set(ylabel="Provinces other than Hubei", xlabel="Values")
china_recent_p_df.drop(['Hong Kong'], inplace= True)

china_recent_p_df.sort_values(['Recovery Rate'])
china_recent_p_df.sort_values(['Mortality Rate'], ascending= False)
wc = r'../input/corona-analysis-files/china.json' # geojson file

tscale= np.linspace(china_recent_p_df['Confirmed'].min(), china_recent_p_df['Confirmed'].max()+1, 6, dtype=int).tolist()



world_map = folium.Map(location=[35.861660, 105.195397], zoom_start= 4, tiles='Mapbox Bright')

world_map.choropleth(

    geo_data= wc,

    data= china_recent_p_df,

    columns=[china_recent_p_df.index, 'Confirmed'],

    key_on='feature.properties.name',

    threshold_scale= tscale,

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Corona Outbreak strength in China'

)



world_map
clatlon= pd.read_csv('../input/corona-analysis-files/China_Provinces_LatLon.csv', index_col= 'Province/State')

clatlon.drop(['Unnamed: 0'], axis= 1, inplace= True)
china_recent_p_df= china_recent_p_df.join(clatlon, how= 'inner')

china_recent_p_df
world_map = folium.Map(location=[35.861660, 110.195397], zoom_start= 5, tiles='Stamen Toner')

outbreaks = folium.map.FeatureGroup()

    

for lt, ln, cd, cnfrm, rec, mor in zip(china_recent_p_df['LAT'], china_recent_p_df['LON'], china_recent_p_df.index, china_recent_p_df['Confirmed'], china_recent_p_df['Recovery Rate'], china_recent_p_df['Mortality Rate']):

    ss= '<b>Province:</b> ' + cd + '<br><b>#Confirmed: </b>' + str(int(cnfrm)) + '<br><b>Recovery rate: </b>' + str(round(rec, 2)) + '<br><b>Mortality rate: </b>' + str(round(mor, 2))

    folium.Marker([lt, ln], popup= ss).add_to(world_map)    

    folium.CircleMarker([lt, ln], radius= 0.05*int(cnfrm), color= 'red').add_to(world_map) 

    

world_map
spread_df= df.groupby(df.index)['Country'].nunique().to_frame()



f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(x= spread_df.index, y= spread_df['Country']).set_title('Number of Countries affected by Corona virus over time', size= 20)

sns.despine(left= True)

ax.set(ylabel="Values", xlabel="Timeline")
over_time_df= df.groupby(df.index).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})

over_time_df['Recovery rate']= over_time_df['Recovered']/over_time_df['Confirmed']

over_time_df['Mortality rate']= over_time_df['Deaths']/over_time_df['Confirmed']



f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(x= over_time_df.index, y=  over_time_df['Confirmed'], label= 'Confirmed', color= 'blue').set_title("#Confirmed Cases, Deaths & Recoveries over time all over the world", size= 20)

sns.lineplot(x= over_time_df.index, y=  over_time_df['Deaths'], label= 'Deaths', color= 'red')

sns.lineplot(x= over_time_df.index, y=  over_time_df['Recovered'], label= 'Recovered', color= 'green')

sns.despine(left= True)

ax.legend(ncol=3, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline", )
f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(x= over_time_df.index, y=  over_time_df['Recovery rate'], label= 'Confirmed', color= 'green').set_title("Mortality rate & Recovery rate over time all over the world", size= 20)

sns.lineplot(x= over_time_df.index, y=  over_time_df['Mortality rate'], label= 'Deaths', color= 'red')

sns.despine(left= True)

ax.legend(ncol=2, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline")
china_over_time_df= df[(df['Country']=='China')]

china_over_time_df.groupby(china_over_time_df.index).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})

np_china_over_time_df= china_over_time_df.groupby(china_over_time_df.index).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})

np_china_over_time_df['Recovery rate']= np_china_over_time_df['Recovered']/np_china_over_time_df['Confirmed']

np_china_over_time_df['Mortality rate']= np_china_over_time_df['Deaths']/np_china_over_time_df['Confirmed']



f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(x= np_china_over_time_df.index, y=  np_china_over_time_df['Confirmed'], label= 'Confirmed', color= 'blue').set_title("#Confirmed Cases, Deaths & Recoveries over time in China", size= 20)

sns.lineplot(x= np_china_over_time_df.index, y=  np_china_over_time_df['Deaths'], label= 'Deaths', color= 'red')

sns.lineplot(x= np_china_over_time_df.index, y=  np_china_over_time_df['Recovered'], label= 'Recovered', color= 'green')

sns.despine(left= True)

ax.legend(ncol=3, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline", )
f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(x= np_china_over_time_df.index, y=  np_china_over_time_df['Recovery rate'], label= 'Confirmed', color= 'green').set_title("Mortality rate & Recovery rate  over time all over the world")

sns.lineplot(x= np_china_over_time_df.index, y=  np_china_over_time_df['Mortality rate'], label= 'Deaths', color= 'red')

sns.despine(left= True)

ax.legend(ncol=2, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline")
china_over_time_df.set_index('Province/State')

china_over_time_df['Mortality rate']= china_over_time_df['Deaths']/china_over_time_df['Confirmed']

china_over_time_df['Recovery rate']= china_over_time_df['Recovered']/china_over_time_df['Confirmed']

china_over_time_df.fillna(0, inplace= True)



f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Hubei')], x= china_over_time_df[(china_over_time_df['Province/State']=='Hubei')].index, y=  'Confirmed', label= 'Hubei', color= 'red').set_title("Comparison of confirmed cases in Hubei vs other Chinese provinces", size= 20)

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Guangdong')], x= china_over_time_df[(china_over_time_df['Province/State']=='Guangdong')].index, y=  'Confirmed', label= 'Guangdong', color= 'green')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Henan')], x= china_over_time_df[(china_over_time_df['Province/State']=='Henan')].index, y=  'Confirmed', label= 'Henan', color= 'blue')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Zhejiang')], x= china_over_time_df[(china_over_time_df['Province/State']=='Zhejiang')].index, y=  'Confirmed', label= 'Zhejiang', color= 'black')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Hunan')], x= china_over_time_df[(china_over_time_df['Province/State']=='Hunan')].index, y=  'Confirmed', label= 'Hunan', color= 'pink')



sns.despine(left= True)

ax.legend(ncol=5, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline", )
f, ax = plt.subplots(figsize=(20, 10))

ax.grid(True)

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Guangdong')], x= china_over_time_df[(china_over_time_df['Province/State']=='Guangdong')].index, y=  'Confirmed', label= 'Guangdong', color= 'green')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Henan')], x= china_over_time_df[(china_over_time_df['Province/State']=='Henan')].index, y=  'Confirmed', label= 'Henan', color= 'blue')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Zhejiang')], x= china_over_time_df[(china_over_time_df['Province/State']=='Zhejiang')].index, y=  'Confirmed', label= 'Zhejiang', color= 'black')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Hunan')], x= china_over_time_df[(china_over_time_df['Province/State']=='Hunan')].index, y=  'Confirmed', label= 'Hunan', color= 'pink')

sns.lineplot(data= china_over_time_df[(china_over_time_df['Province/State']=='Anhui')], x= china_over_time_df[(china_over_time_df['Province/State']=='Anhui')].index, y=  'Confirmed', label= 'Anhui 	', color= 'red').set_title("Confirmed cases over time in Chinese provinces except Hunan", size= 20)



sns.despine(left= True)

ax.legend(ncol=5, loc="upper left")

ax.set(ylabel="Values", xlabel="Timeline", )