import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

import fiona

from datetime import datetime



from ipywidgets import SelectionSlider, interact

import geopandas as gpd



import shapely
# enable KML support which is disabled by default

fiona.drvsupport.supported_drivers['kml'] = 'rw'

fiona.drvsupport.supported_drivers['KML'] = 'rw'
DEFAULT_FIGSIZE = (30, 10)

DEFAULT_CRS = "EPSG3857"
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'), crs=DEFAULT_CRS)
def show_on_earth(*dfs):

    ax = world.plot(figsize=DEFAULT_FIGSIZE, edgecolor='gray', color="white")

    colors = ["blue", "green", "red", "orange", "tomato", "lime"]

    for color, df in zip(colors, dfs):

        df.plot(ax=ax,color=color)
urls = [

 'al052019_5day_001.zip',

 'al052019_5day_002.zip',

 'al052019_5day_003.zip',

 'al052019_5day_004.zip',

 'al052019_5day_004A.zip',

 'al052019_5day_005.zip',

 'al052019_5day_005A.zip',

 'al052019_5day_006.zip',

 'al052019_5day_006A.zip',

 'al052019_5day_007.zip',

 'al052019_5day_007A.zip',

 'al052019_5day_008.zip',

 'al052019_5day_008A.zip',

 'al052019_5day_009.zip',

 'al052019_5day_009A.zip',

 'al052019_5day_010.zip',

 'al052019_5day_010A.zip',

 'al052019_5day_011.zip',

 'al052019_5day_011A.zip',

 'al052019_5day_012.zip',

 'al052019_5day_012A.zip',

 'al052019_5day_013.zip',

 'al052019_5day_013A.zip',

 'al052019_5day_014.zip',

 'al052019_5day_014A.zip',

 'al052019_5day_015.zip',

 'al052019_5day_015A.zip',

 'al052019_5day_016.zip',

 'al052019_5day_016A.zip',

 'al052019_5day_017.zip',

 'al052019_5day_017A.zip',

 'al052019_5day_018.zip',

 'al052019_5day_018A.zip',

 'al052019_5day_019.zip',

 'al052019_5day_020.zip',

 'al052019_5day_021.zip',

 'al052019_5day_022.zip',

 'al052019_5day_023.zip',

 'al052019_5day_024.zip',

 'al052019_5day_024A.zip',

 'al052019_5day_025.zip',

 'al052019_5day_025A.zip',

 'al052019_5day_026.zip',

 'al052019_5day_026A.zip',

 'al052019_5day_027.zip',

 'al052019_5day_027A.zip',

 'al052019_5day_028.zip',

 'al052019_5day_028A.zip',

 'al052019_5day_029.zip',

 'al052019_5day_029A.zip',

 'al052019_5day_030.zip',

 'al052019_5day_030A.zip',

 'al052019_5day_031.zip',

 'al052019_5day_031A.zip',

 'al052019_5day_032.zip',

 'al052019_5day_032A.zip',

 'al052019_5day_033.zip',

 'al052019_5day_033A.zip',

 'al052019_5day_034.zip',

 'al052019_5day_034A.zip',

 'al052019_5day_035.zip',

 'al052019_5day_035A.zip',

 'al052019_5day_036.zip',

 'al052019_5day_036A.zip',

 'al052019_5day_037.zip',

 'al052019_5day_037A.zip',

 'al052019_5day_038.zip',

 'al052019_5day_038A.zip',

 'al052019_5day_039.zip',

 'al052019_5day_039A.zip',

 'al052019_5day_040.zip',

 'al052019_5day_040A.zip',

 'al052019_5day_041.zip',

 'al052019_5day_041A.zip',

 'al052019_5day_042.zip',

 'al052019_5day_042A.zip',

 'al052019_5day_043.zip',

 'al052019_5day_043A.zip',

 'al052019_5day_044.zip',

 'al052019_5day_044A.zip',

 'al052019_5day_045.zip',

 'al052019_5day_045A.zip',

 'al052019_5day_046.zip',

 'al052019_5day_046A.zip',

 'al052019_5day_047.zip',

 'al052019_5day_047A.zip',

 'al052019_5day_048.zip',

 'al052019_5day_048A.zip',

 'al052019_5day_049.zip',

 'al052019_5day_049A.zip',

 'al052019_5day_050.zip',

 'al052019_5day_050A.zip',

 'al052019_5day_051.zip',

 'al052019_5day_051A.zip',

 'al052019_5day_052.zip',

 'al052019_5day_052A.zip',

 'al052019_5day_053.zip',

 'al052019_5day_053A.zip',

 'al052019_5day_054.zip',

 'al052019_5day_054A.zip',

 'al052019_5day_055.zip',

 'al052019_5day_055A.zip',

 'al052019_5day_056.zip',

 'al052019_5day_056A.zip',

 'al052019_5day_057.zip',

 'al052019_5day_058.zip',

 'al052019_5day_059.zip',

 'al052019_5day_059A.zip',

 'al052019_5day_060.zip',

 'al052019_5day_060A.zip',

 'al052019_5day_061.zip',

 'al052019_5day_061A.zip',

 'al052019_5day_062.zip',

 'al052019_5day_062A.zip',

 'al052019_5day_063.zip',

 'al052019_5day_063A.zip',

 'al052019_5day_064.zip'

]
base = "https://www.nhc.noaa.gov/gis/forecast/archive"
hurrs = [gpd.read_file(f"{base}/{url}") for url in urls]
hurricane_history = pd.concat(hurrs)

hurricane_history.geometry = hurricane_history.geometry.apply(shapely.geometry.Polygon)

hurricane_history = hurricane_history[hurricane_history["STORMTYPE"] == "HU"]
hurricane_history.head()
hurricane_history["date_time"] = hurricane_history["ADVDATE"].apply(lambda d: datetime.strptime(d[:1]+" "+d[1:3] +" "+" "+d[-11:], "%H %M %b %d %Y"))

hurricane_history = hurricane_history.set_index('date_time')

hurricane_history = hurricane_history.sort_index()
hurricane_history = hurricane_history[hurricane_history["STORMTYPE"] == 'HU']

hurricane_history = hurricane_history[~hurricane_history.index.duplicated(keep='first')]
canada = gpd.read_file(

    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson",

    crs=DEFAULT_CRS

)
states = gpd.read_file(

    "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json",

    crs=DEFAULT_CRS,

)

states = states.rename(columns={"NAME": "name"})
north_pacific = pd.concat([canada, states])
north_pacific.head()
show_on_earth(north_pacific)
event_date_selector = SelectionSlider(

    options=hurricane_history.index,

    value=hurricane_history.index[30]

)

@interact(event_date=event_date_selector, continous_update=False)

def show_timelapse(event_date):

    h = hurricane_history[hurricane_history.index==event_date]

    print(f"Calculating effected areas... {event_date}")

    effected_areas = north_pacific[north_pacific.intersects(h.unary_union)]

    print(f"{len(effected_areas)} areas effected by the hurricane")

    for location in effected_areas.itertuples():

        damage = (location.geometry.intersection(h.unary_union).area / location.geometry.area) * 100

        print(f"{damage:.2f}% of {location.name}")

    

    show_on_earth(h, north_pacific, effected_areas)
