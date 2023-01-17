import requests
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import re
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
data = pd.read_csv('../input/professional-location-cordinates/litlong.csv')
data.tail(5)
data.dtypes
lat1   = 28.677212    #Customer Latitude 
lon1  = 77.500419     #customer Longitude
Customer_city_name = 'modinagar'
booleans = []
for city in data.city:
    if city == Customer_city_name:
        booleans.append(True)
    else:
        booleans.append(False)
booleans[0:5] #first five row if you want see all just type 'booleans'
your_city_list = pd.Series(booleans)
your_city_list.head(10)
city_sort_data = data[your_city_list] #pandas dataframe which only contain customer city
city_sort_data[0:]
vector_dis_list = []
for a, b in zip(city_sort_data.iloc[:, 1], city_sort_data.iloc[:, 2]):
    lat2 = float(format(a, '.8g'))
    lon2 = float(format(b, '.8g'))
    print(lat2, lon2)
    
    distance = haversine(lon1, lat1, lon2, lat2)
    vector_dis_list.append(distance)
    
vector_dis_list = pd.DataFrame(vector_dis_list)

vector_dis_list
vector_dis_list = vector_dis_list.rename(columns={ 0 : 'vector_distance'})
vector_dis_list
city_sort_data.reset_index(drop=True, inplace=True)
vector_dis_list.reset_index(drop=True, inplace=True)
vector_dis_data = pd.concat([city_sort_data, vector_dis_list], axis=1)
vector_dis_data
vector_dis_data.sort_values( 'vector_distance' , axis = 0, ascending = True, 
                 inplace = True, na_position ='last')
vector_dis_data_n = vector_dis_data.head(8) #assume we are only intresting for nearest 8 professionals business
vector_dis_data_n
vector_dis_data_n.reset_index(drop=True, inplace=True)
vector_dis_data_n
def actual_distence(lat1, lon1, lat2, lon2):


    url = "https://distance-calculator.p.rapidapi.com/distance/simple"

    querystring = {"unit":"kilometers","lat_1": lat1 ,"long_2": lon2 ,"long_1": lon1 , "lat_2": lat2}

    headers = {
        'x-rapidapi-host': "distance-calculator.p.rapidapi.com",
        'x-rapidapi-key': "0115868787msh5e40d384c701b30p156c33jsn521c6b06043b",
        'content-type': "application/json"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    return response.text
final_data = []
for a, b in zip(vector_dis_data_n.iloc[:, 1], vector_dis_data_n.iloc[:, 2]):
    lat2 = float(format(a, '.8g'))
    lon2 = float(format(b, '.8g'))

    final_distance = actual_distence(lat1, lon1, lat2, lon2)
    print(final_distance)
    result = re.findall(r"[-+]?\d*\.\d+|\d+", final_distance) #for extracting float value from the string
    
    final_data.append(result)
final_data
final_data = pd.DataFrame(final_data)
final_data
modified_list= []
for i in final_data[0]:
    update = float(i)
    print(update)
    if update <= 1:
        distance = int(1000 * update)
        distance = str(distance) + ' ' +'meter'
    else:
        distance = str(format(update, '.2f')) + ' ' + 'Km'
        

    modified_list.append(distance)
modified_list = pd.DataFrame(modified_list)
modified_list.head(10)
final_modified_list = []
for i in modified_list[0]:
    distance = (i)
    final_modified_list.append(distance)
final_modified_list = pd.DataFrame(final_modified_list)
final_modified_list
final_modified_list = final_modified_list.rename(columns={ 0 : 'distance'})
PRO_FINAL = pd.concat([vector_dis_data_n, final_modified_list], axis=1)
PRO_FINAL
suggested_list = []
for i in PRO_FINAL['Professionals_id']:
    suggested_list.append(i)
suggested_list
