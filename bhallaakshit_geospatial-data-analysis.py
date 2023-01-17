from geopy.geocoders import Nominatim # To obtain coordinates and address given the name
import folium # For making maps
from folium.plugins import MarkerCluster # To make maps
import os # For input-output operations
import pandas as pd # For DataFrame operations
# Obtaining files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
excel = pd.ExcelFile("../input/commercial-areas-in-bangalore/commercial.xlsx")
def GetData(name, excel):
    # Read sheet in file
    df = excel.parse(name, header = None)
    # Preprocess
    if name == "Malls":
        df[0] += " Mall"
    
    AddList = []
    LatList = []
    LonList = []
    
    # Define geolocator 
    geolocator = Nominatim(user_agent = "Akshit Bhalla",timeout = 1000)
    # Obtain required
    for item in df[0]:
        location = geolocator.geocode(str(item) + ", Bangalore")
        
        AddList.append(location.address)
        LatList.append(location.latitude) 
        LonList.append(location.longitude)
    
    return(list(df[0].values), LatList, LonList, AddList)
malls, mall_lat, mall_lon, mall_add = GetData("Malls", excel)
restaurants, restaurant_lat, restaurant_lon, restaurant_add = GetData("Restaurants", excel)
offices, office_lat, office_lon, office_add = GetData("Offices", excel)
# Define map
MyMap = folium.Map(
    location = (12.9716, 77.5946), # Coordinates of Bangalore
    zoom_start = 10
)
# Function to make elements to be displayed on the map
def MakeMap(name, cat, lat, lon, add, mc):
    if name == "Mall":
        color = "blue"
    elif name == "Restaurant":
        color = "green"
    else:
        color = "red"
    
    for i in range(35):
        mc.add_child(
            folium.Marker(
                location = (lat[i], lon[i]),
                popup = (
                    "<b>" + name + ":</b> {}<br>"
                    "<b>Address:</b> {}"
                ).format(cat[i], add[i]),
                icon = folium.Icon(color = color)
            )
        )
        
    return mc
mc1 = MakeMap("Mall", malls, mall_lat, mall_lon, mall_add, MarkerCluster())
mc2 = MakeMap("Restaurant", restaurants, restaurant_lat, restaurant_lon, restaurant_add, MarkerCluster())
mc3 = MakeMap("Office", offices, office_lat, office_lon, office_add, MarkerCluster())
# Create groups of features
fg1 = folium.FeatureGroup(name = "malls")
fg2 = folium.FeatureGroup(name = "restraunts")
fg3 = folium.FeatureGroup(name = "offices")
# Add elements to respective groups
mc1.add_to(fg1)
mc2.add_to(fg2)
mc3.add_to(fg3)
# Add groups to map
MyMap.add_child(fg1)
MyMap.add_child(fg2)
MyMap.add_child(fg3)
# Display map with layers
MyMap.add_child(folium.LayerControl())
# Save Map as HTML
MyMap.save("MyMap.html")
# Create DataFrame of necessary data
MyData = pd.DataFrame({
    "Mall" : malls,
    "Latitude" : mall_lat,
    "Longitude" : mall_lon,
    "Address" : mall_add,
    "Restaurant" : restaurants,
    "Latitude" : restaurant_lat,
    "Longitude" : restaurant_lon,
    "Address" : restaurant_add,
    "Office" : offices,
    "Latitude" : office_lat,
    "Longitude" : office_lon,
    "Address" : office_add
})
# Export DataFrame
MyData.to_csv("Commercial_Data.csv") 