!pip install geopandas

!pip install folium

!pip install shapely

!pip install unidecode
import json

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt

import shapely

import folium

from folium.plugins import MarkerCluster,FastMarkerCluster, HeatMap

import base64

import random

from unidecode import unidecode
def _repr_html_(self, **kwargs):

    html = base64.b64encode(self.render(**kwargs).encode('utf8')).decode('utf8')

    onload = (

        'this.contentDocument.open();'

        'this.contentDocument.write(atob(this.getAttribute(\'data-html\')));'

        'this.contentDocument.close();'

    )

    if self.height is None:

        iframe = (

            '<div style="width:{width};">'

            '<div style="position:relative;width:100%;height:0;padding-bottom:{ratio};">'

            '<iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;'

            'border:none !important;" '

            'data-html={html} onload="{onload}" '

            'allowfullscreen webkitallowfullscreen mozallowfullscreen>'

            '</iframe>'

            '</div></div>').format

        iframe = iframe(html=html, onload=onload, width=self.width, ratio=self.ratio)

    else:

        iframe = ('<iframe src="about:blank" width="{width}" height="{height}"'

                  'style="border:none !important;" '

                  'data-html={html} onload="{onload}" '

                  '"allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen">'

                  '</iframe>').format

        iframe = iframe(html=html, onload=onload, width=self.width, height=self.height)

    return iframe



folium.branca.element.Figure._repr_html_ = _repr_html_
with open("../input/istanbul-municipality-bus-stops-dataset/durak_datasi.json", "r") as f:

    durak_data = json.load(f)





df = pd.DataFrame(durak_data)

len(df)
df.columns
df["ILCEADI"].unique()
df["SYON"].unique()
df["DURAK_TIPI"].unique()
df["FIZIKI"].unique()
df["AKILLI"].unique()
gdf = gpd.GeoDataFrame(df, geometry=[shapely.wkt.loads(x) for x in df["KOORDINAT"]])

gdf = gdf.drop(columns=["KOORDINAT"])

def create_map():

    return folium.Map(location=[41.0082, 28.9784], zoom_level=15, tiles="cartodbpositron")
durak_map = create_map()



for i,g in enumerate(gdf.iterrows()):

    g = g[1] # get the row

    name = g["SDURAKADI"]

    if name == None:

        name = "Adsız"

        

    folium.vector_layers.CircleMarker(location=(g.geometry.y, g.geometry.x), radius=3, 

                                      color="darkblue", fill=True, stroke=True, popup=unidecode(name)).add_to(durak_map)

    

durak_map
isi_map = create_map()

all_markers = [(m.y, m.x) for m in gdf.geometry]



isi_map.add_child(HeatMap(all_markers, radius=5, blur=2))

isi_map
ilce_durak_map = create_map()



# associate every region with a separate color

regions = df["ILCEADI"].unique().tolist()

colors  = ["#{:03x}".format(random.randint(0, 0xFFF)) for x in regions] # hex colors



colors_dict = dict(zip(regions,colors))



for i,g in enumerate(gdf.iterrows()):

    g = g[1]

    c = colors_dict[g["ILCEADI"]]

    folium.vector_layers.CircleMarker(location=(g.geometry.y, g.geometry.x), radius=3, 

                                      color=c, fill=True, stroke=True, popup=unidecode(g["ILCEADI"])).add_to(ilce_durak_map)



ilce_durak_map
ilce_durak_map = create_map()

grp = gdf.groupby(gdf["ILCEADI"])



for g in grp:

    ilce_name = g[0]

    ilce_df = g[1]

    

    mc = MarkerCluster(name=ilce_name)

    for point in ilce_df.iterrows():

        point = point[1] # get the series

        mc.add_child(folium.Marker(location=(point.geometry.y, point.geometry.x), 

                                   popup=unidecode(f"{point['SDURAKADI']}, {ilce_name}")))

    ilce_durak_map.add_child(mc)



folium.LayerControl(hideSingleBase=True).add_to(ilce_durak_map)

ilce_durak_map
tip_durak_map = create_map()



types = df["DURAK_TIPI"].unique().tolist()

colors  = ["#{:03x}".format(random.randint(0, 0xFFF)) for x in regions] # hex colors



colors_dict = dict(zip(types,colors))

feat_groups = dict(zip(types, [folium.FeatureGroup(name=t) for t in types]))





for i,g in enumerate(gdf.iterrows()):

    g = g[1]

    c = colors_dict[g["DURAK_TIPI"]]

    cm = folium.vector_layers.CircleMarker(location=(g.geometry.y, g.geometry.x), 

                                      radius=3, color=c, fill=True, stroke=True, 

                                      popup=unidecode(g["DURAK_TIPI"]))

    feat_groups[g["DURAK_TIPI"]].add_child(cm)



for f in feat_groups.values():

    tip_durak_map.add_child(f)

    

folium.LayerControl(hideSingleBase=True).add_to(tip_durak_map)

tip_durak_map
mesafe_map = create_map()



for i,g in enumerate(gdf.iterrows()):

    g = g[1]

    folium.vector_layers.Circle(location=(g.geometry.y, g.geometry.x), radius=250, 

                                color="lightgreen", fill=True, stroke=False, fillOpacity=0.1).add_to(mesafe_map)



mesafe_map


