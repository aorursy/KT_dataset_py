#Libraries Needed

import pandas as pd



#Settings to display pandas

pd.set_option('display.max_columns', None)



#Some basic set up

base_path = "/kaggle/input/uncover/UNCOVER/"



#Using the Coders Against COVID as example

CAC_path = base_path + "coders_against_covid/"

CAC_file = CAC_path + "crowd-sourced-covid-19-testing-locations.csv"

CAC_df = pd.read_csv(CAC_file)

#Get one row as example

CAC_df.head(1)
import folium

from folium import plugins



def simple_folium(df:pd.DataFrame, lat_col:str, lon_col:str, text_cols:list, map_name:str):

    """

    Descrption

    ----------

        Returns a simple Folium HeatMap with Markers

    ----------

    Parameters

    ----------

        df : padnas DataFrame, required

            The DataFrane with the data to map

        lat_col : str, required

            The name of the column with latitude

        lon_col : str, required

            The name of the column with longitude

        test_cols: list, optional

            A list with the names of the columns to print for each marker



    """

    #Preprocess

    #Drop rows that do not have lat/lon

    df = df[df[lat_col].notnull() & df[lon_col].notnull()]



    # Convert lat/lon to (n, 2) nd-array format for heatmap

    # Then send to list

    df_locs = list(df[[lat_col, lon_col]].values)



    # Add the location name to the markers

    text_cols = ["location_name", "location_contact_url_main","location_address_street"]

    text_feature_list = list(zip(*[df[col] for col in text_cols]))

    text_formated = []

    for text in text_feature_list:

        text = [str(feat) for feat in text]

        text_formated.append("<br>".join(text))

    marker_info = text_formated



    #Set up folium map

    fol_map = folium.Map([41.8781, -87.6298], zoom_start=4)



    # plot heatmap

    heat_map = plugins.HeatMap(df_locs, name=map_name)

    fol_map.add_child(heat_map)



    # plot markers

    markers = plugins.MarkerCluster(locations = df_locs, popups = marker_info, name="Testing Site")

    fol_map.add_child(markers)



    #Add Layer Control

    folium.LayerControl().add_to(fol_map)



    return fol_map



simple_folium(CAC_df, "lat", "lng", ["location_name","location_address_street"], "COVID Testing Sites")
from simple_folium import simple_folium

help(simple_folium)
simple_folium(df=CAC_df.head(100), lat_col="lat", lon_col="lng", 

              text_cols=["location_name","location_address_street"], 

              map_name="COVID Testing Sites")