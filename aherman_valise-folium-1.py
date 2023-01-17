import folium
import json

FICHIER_COMMUNES = '/kaggle/input/communes_4326_s5pc_hdf.geojson'

def creation_carto():    

    # Récupération du json directement à partir d'un fichier (génération avec QGIS par exemple)
    communes=json.load(open(FICHIER_COMMUNES,encoding='utf-8'))
   # Création de la carte avec point de départ, zoom de départ, et zoom max
    m = folium.Map(location=[49.994, 2.873], 
                   zoom_start=9,
                   max_zoom=14)    
    # Création de la couche 
    folium.GeoJson(
                        communes,
                        name='Communes'
                      ).add_to(m)
    folium.LayerControl().add_to(m)    
    return m
    #m.save(fichier_html)

creation_carto()