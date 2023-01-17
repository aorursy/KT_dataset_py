# Install the Lets-Plot library
!pip install lets-plot -U
from lets_plot import *

LetsPlot.setup_html(show_status=True)
from lets_plot.geo_data import *

ports_of_embarkation = ['Southampton', 'Cherbourg', 'Cobh']
# Obtain the 'Regions' object containing IDs of: Southampton, Cherbourg and Cobh regions (cities).
ports_of_embarkation_geocoded = regions_builder(level='city', request=ports_of_embarkation) \
        .where('Cherbourg', within='France') \
        .where('Southampton', within='England') \
        .build()
ports_of_embarkation_geocoded
LetsPlot.set(maptiles_zxy(url='https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}@2x.png'))

basemap = ggplot() + ggsize(800, 300) \
    + geom_livemap(map=ports_of_embarkation_geocoded,
                   size=7, 
                   shape=21, color='black', fill='yellow')

basemap
from shapely.geometry import Point, LineString
titanic_site = Point(-38.056641, 46.920255)

# Add the marker using the `geom_point` geometry layer.
titanic_site_marker = geom_point(x=titanic_site.x, y = titanic_site.y, size=10, shape=9, color='red')

basemap + titanic_site_marker
from geopandas import GeoSeries
from geopandas import GeoDataFrame

# The points of embarkation
embarkation_points = ports_of_embarkation_geocoded.centroids().geometry
titanic_journey_points = embarkation_points.append(GeoSeries(titanic_site), ignore_index=True)

# New GeoDataFrame containing a `LineString` geometry.
titanic_journey_gdf = GeoDataFrame(dict(geometry=[LineString(titanic_journey_points)]))

# App the polyline using the `geom_path` layer.
titanic_path = geom_path(map=titanic_journey_gdf, color='dark-blue', linetype='dotted', size=1.2)

basemap + titanic_path + titanic_site_marker
# Geocoding of The New York City is a trivial task.
NYC = regions_city(['New York']).centroids().geometry[0]

map_layers = titanic_path \
  + geom_segment(x=titanic_site.x, y=titanic_site.y, 
                 xend=NYC.x, yend=NYC.y, 
                 color='white', linetype='dotted', size=1.2) \
  + geom_point(x=NYC.x, y=NYC.y, size=7, shape=21, color='black', fill='white') \
  + titanic_site_marker

basemap + map_layers
import pandas as pd
df = pd.read_csv("../input/titanic-cleaned-data/train_clean.csv")
df.head()
from lets_plot.mapping import as_discrete

bars = ggplot(df) \
    + geom_bar(aes('Embarked', fill=as_discrete('Survived')), position='dodge') \
    + scale_fill_discrete(labels=['No', 'Yes']) \
    + scale_x_discrete(labels=['Southampton', 'Cobh', 'Cherbourg'], limits=['S', 'C', 'Q'])

bars + ggsize(800, 250)
bars_settings = theme(axis_title='blank', 
                   axis_line='blank', 
                   axis_ticks_y='blank',
                   axis_text_y='blank',
                   legend_position=[1.12, 1.07],
                   legend_justification=[1, 1]) + scale_x_discrete(expand=[0, 0.05])


basemap = ggplot() + ggsize(800, 300) \
    + geom_livemap(map=ports_of_embarkation_geocoded.centroids(), 
                    size=8, 
                    shape=21, color='black', fill='yellow',
                    zoom=4, location=[-12, 48])

fig = GGBunch()
fig.add_plot(basemap + map_layers, 0, 0)
fig.add_plot(bars + bars_settings, 535, 135, 250, 150)
fig