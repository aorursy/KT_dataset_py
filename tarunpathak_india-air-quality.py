#importing libraries
import pandas as pd, folium
from folium.plugins import HeatMap, HeatMapWithTime
from google.cloud import bigquery
#latest data for India
#showing concentration of pollutants over cities
query ="""select * from 
(
    select row_number() over(partition by q.timestamp,
    q.location, q.pollutant order by q.timestamp desc) as rnum, q.* 
    from bigquery-public-data.openaq.global_air_quality q
    where extract(year FROM q.timestamp)=extract(year from current_date())
    and q.country='IN' 
) x where x.rnum=1"""

client = bigquery.Client()
job = client.query(query)
df = job.to_dataframe()

#plotting air quality data
#using heatmap
f=folium.Figure(width=1500,height=800)
m = folium.Map(location=[21.1458,79.0882], tiles='openstreetmap', zoom_start=5)

pollutants={'co':'Carbon Monoxide', 'no2':'Nitrogen Dioxide','so2':'Sulphar Dioxide','pm10':'Particulate Matter 10', 'pm25':'Particulate Matter 2.5'}
for p in pollutants.keys():
    filtered_df=df[df['pollutant']==p]
    fg=folium.FeatureGroup(name=pollutants[p])
    HeatMap(data=filtered_df[['latitude','longitude']],radius=12).add_to(fg)
    fg.add_to(m)

#displaing map
folium.LayerControl().add_to(m)
f.add_child(m)
f
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')
#getting data
query ="""select q.* from bigquery-public-data.openaq.global_air_quality q
    where extract(year FROM q.timestamp) in (2019,2020)
    and q.country='IN'"""

client = bigquery.Client()
job = client.query(query)
df = job.to_dataframe()

#converting timestamp to date
df['timestamp'] = df['timestamp'].values.astype(dtype='datetime64[D]')

#displaying seperate map for each pollutant
pollutants={'co':'Carbon Monoxide', 'no2':'Nitrogen Dioxide','so2':'Sulphar Dioxide','pm10':'Particulate Matter 10', 'pm25':'Particulate Matter 2.5'}
for p in pollutants.keys():
    
    #filtering data
    filtered_df=df[df['pollutant']==p]
    
    #processing data
    data=[]
    for _, d in filtered_df.groupby('timestamp'):
        data.append([[row['latitude'], row['longitude'], row['value']] for _, row in d.iterrows()])
    
    #displaying map
    f = folium.Figure(width=1500,height=800)
    m = folium.Map(location=[21.1458,79.0882], tiles='openstreetmap', zoom_start=5)
    
    hm = HeatMapWithTime(data, auto_play=True, max_opacity=0.8).add_to(m)
    
    f.add_child(m)
    embed_map(f,'map_covid.html')