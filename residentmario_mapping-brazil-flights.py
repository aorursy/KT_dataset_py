import geoplot as gplt
import pandas as pd
import numpy as np

df = pd.read_csv('../input/flights-in-brazil/BrFlights2/BrFlights2.csv', encoding='latin1')
df.columns = ['Flights', 'Airline', 'Flight_Type','Departure_Estimate','Departure_Real','Arrival_Estimate','Arrival_Real','Flight_Situation','Code_Justification','Origin_Airport','Origin_City','Origin_State','Origin_Country','Destination_Airport','Destination_City','Destination_State','Destination_Country','Destination_Long','Destination_Lat','Origin_Long','Origin_Lat']
df.head()
df_airports = pd.DataFrame()
df_airports['name'] = df['Origin_Airport'].unique()
df_airports['Lon'] = df['Origin_Long'].unique()
df_airports['Lat'] = df['Origin_Lat'].unique()
df_path = df[['Destination_Long', 'Destination_Lat','Origin_Long','Origin_Lat']]
df_path = df_path.drop_duplicates()
df_path = df_path.reset_index()
AirPortCount = pd.DataFrame(df.groupby(by=['Origin_Airport'])['Departure_Estimate'].count().reset_index())
AirPortCount.columns = ['Origin_Airport','Count']
df_airports = df_airports.merge(AirPortCount,left_on='name',right_on='Origin_Airport')
airports = [ dict(
        type = 'scattergeo',
        lon = df_airports['Lon'],
        lat = df_airports['Lat'],
        hoverinfo = 'text',
        text = df_airports['name'],
        mode = 'markers',
        marker = dict( 
            size=np.log10(df_airports['Count'])*1.35, 
            color='rgb(255, 0, 0)',
            line = dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        ))]
flight_paths = []
for i in range(len(df_path)):
    flight_paths.append(dict(
            type = 'scattergeo',
            lon = [df_path['Origin_Long'][i], df_path['Origin_Lat'][i]],
            lat = [df_path['Destination_Long'][i], df_path['Destination_Lat'][i]],
            mode = 'lines',
            line = dict(
                width = 0.1,
                color = 'red',
            ),
            opacity = 0.5,
        ))
from shapely.geometry import Point

connections = (
    pd.DataFrame(flight_paths)
        .rename(columns={'lat': 'origin', 'lon': 'destination'})
        .drop(['line', 'mode', 'opacity', 'type'], axis='columns')
        .pipe(lambda df: df.assign(origin=[Point(*p) for p in df.origin], destination=[Point(*p) for p in df.destination]))
)
import geopandas as gpd
shp = gpd.read_file("../input/countries-shape-files/ne_10m_admin_0_countries.shp")
connections
import geoplot.crs as gcrs

ax = gplt.polyplot(shp, projection=gcrs.Robinson(), linewidth=0, figsize=(24, 12), color='lightgray')
gplt.sankey(connections, start='origin', end='destination', projection=gcrs.Robinson(), ax=ax, color='black')
ax.set_global()