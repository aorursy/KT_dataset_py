import json

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt

import shapely

import folium

from folium.plugins import MarkerCluster,FastMarkerCluster, HeatMap, Fullscreen

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
gdf_districts = gpd.read_file("../input/turkey-districts/tur_polbnda_adm2.shp")

gdf_districts = gdf_districts[gdf_districts["adm1_en"] == "ISTANBUL"]
with open("../input/istanbul-municipality-bus-stops-dataset/durak_datasi.json", "r") as f:

    durak_data = json.load(f)



df = pd.DataFrame(durak_data)
gdf = gpd.GeoDataFrame(df, geometry=[shapely.wkt.loads(x) for x in df["KOORDINAT"]])

gdf = gdf.drop(columns=["KOORDINAT"])
def create_map():

    return folium.Map(location=[41.0082, 28.9784], zoom_level=15, tiles="cartodbpositron")
durak_map = create_map()



ilce_kume_fg = folium.FeatureGroup(name="Durak Kümeleri")

ilce_sinir_fg = folium.FeatureGroup(name="İlçe Sınırları")

isi_fg = folium.FeatureGroup(name="Isı haritası")



all_markers = [(m.y, m.x) for m in gdf.geometry]

isi_fg.add_child(HeatMap(all_markers, radius=5, blur=2))





grp = gdf.groupby(gdf["ILCEADI"])



for g in grp:

    ilce_name = g[0]

    ilce_df = g[1]

    

    mc = MarkerCluster(name=ilce_name)

    for point in ilce_df.iterrows():

        point = point[1] # get the series

        durak_adi = point['SDURAKADI']

        if durak_adi is None:

            durak_adi = "Adsiz"

        durak_tipi = point["DURAK_TIPI"]

        if durak_tipi is None:

            durak_tipi = "Belirtilmemis"

        fiziki = point["FIZIKI"]

        if fiziki is None:

            fiziki = "Belirtilmemis."

        popup_html = f"{unidecode(durak_adi)}       <hr><b>Durak tipi:</b> {unidecode(durak_tipi)}<br><b>Akilli:</b> {point['AKILLI']}<br><b>Fiziki:</b> {unidecode(fiziki)}"

        mc.add_child(folium.Marker(location=(point.geometry.y, point.geometry.x),popup=popup_html))

    ilce_kume_fg.add_child(mc)





for p in gdf_districts.iterrows():

    p = p[1]

    folium.GeoJson(p.geometry, tooltip=p.adm2_en).add_to(ilce_sinir_fg)



durak_map.add_child(ilce_kume_fg)

durak_map.add_child(ilce_sinir_fg)

durak_map.add_child(isi_fg)



lc = folium.LayerControl(hideSingleBase=True)

lc.add_child(Fullscreen())

durak_map.add_child(lc)

durak_map