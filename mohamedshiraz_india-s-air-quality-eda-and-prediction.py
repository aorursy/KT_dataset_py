# loading in some libraries 



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium 

import plotly.express as px



%matplotlib inline



# Geopy is used to find geo-data of street addresses and states. 

import geopy

from geopy.geocoders import Nominatim



import warnings

warnings.filterwarnings('ignore')

#inputting the data 



df_city = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')

df_stations = pd.read_csv('/kaggle/input/air-quality-data-in-india/station_day.csv')

stations = pd.read_csv('../input/air-quality-data-in-india/stations.csv')

print(df_city.info())

df_city.head()

print(df_stations.info())



df_stations.head()
print(stations.info())



stations.head()
df_stations = pd.merge(df_stations,stations,how = 'left')

df_stations.head()

#That's better
df_city.isnull().sum().plot(kind ='bar')
df_stations.isnull().sum().plot(kind ='bar')
df_city['AQI'].fillna(method = 'ffill',inplace = True)

df_city['AQI'].fillna(method = 'bfill',inplace = True)

df_city['AQI'].fillna(value = 0,inplace = True)





df_stations['AQI'].fillna(method = 'ffill',inplace = True)

df_stations['AQI'].fillna(method = 'bfill',inplace = True)

df_stations['AQI'].fillna(value = 0,inplace = True)



# df_stations['AQI_Bucket'].fillna('Not Provided',inplace = True)


# Finds the geo coordinates of the stations

def get_geo(station_address,city):

    # getting just the street address from station_address

    

    street_address = ''

    temp = station_address.split(',')

    street_address = temp[:1]

        

    geolocator = Nominatim(user_agent="Shirazfromkaggle")

    location = geolocator.geocode(street_address)

    

    if location is None:

        #If street level address cant be found, then just the city's coordinates should do

        location = geolocator.geocode(city)

        

        if location is None :

            #If the coordiantes of the city cant be found either, then return None.

            location = [None,None]

            return location

    

    return [location.latitude,location.longitude],street_address





# Finds the street-level addresses of the stations 

def get_address(station_address):

    street_address = ''

    temp = station_address.split()

    for i in range (0,len(temp)):

                    

        street_address += temp[i]

        if temp[i][-1] == ',':

            street_address += (" "+temp[i+1])

            break

        else:

            street_address+=' '

    return street_address



#Finds the AQI buckets from inputted AQI.

def get_AQI_bucket(x):

    

    # Here, x = AQI

    if x <= 50:

        return "Good"

    elif x <= 100:

        return "Satisfactory"

    elif x <= 200:

        return "Moderate"

    elif x <= 300:

        return "Poor"

    elif x <= 400:

        return "Very Poor"

    elif x > 400:

        return "Severe"

    else:

        return np.NaN
import json 



# geojson for the choropleth map 



india_states = json.load(open('../input/geojson/india_states.geojson','r'))



state_id_map = {}



for features in india_states['features']:

    features['id'] = features['properties']['ID_1']

    state_id_map[features['properties']['NAME_1']] = features['id']

    

state_id_map
# def get_state(city):

#     # getting just the state in which the city is present

#     geolocator = Nominatim(user_agent="Shirazfromkaggle")

#     location = geolocator.geocode(city)

    

#     if location is None:

#         return 'Nan'

#     else:

#         state = location.address.split(sep = ',')[-2]

#         # location.address.split displays address as ['சென்னை - Chennai', ' Chennai district', ' Tamil Nadu', ' India']



#         if state.lstrip().isdigit():

#             state = location.address.split(sep = ',')[-3]

        

#         return state.lstrip()



# l = df_city['City'].unique()



# city_states ={}

# for i in l:

#     city_states[i] = get_state(i)

# city_states

# # df_city['state'] = df_city['City'].apply(get_state)

# # df_city['state'].unique()
city_states = {'Ahmedabad': 'Gujarat',

 'Aizawl': 'Mizoram',

 'Amaravati': 'Andhra Pradesh',

 'Amritsar': 'Punjab',

 'Bengaluru': 'Karnataka',

 'Bhopal': 'Madhya Pradesh',

 'Brajrajnagar': 'Orissa',

 'Chandigarh': 'Chandigarh',

 'Chennai': 'Tamil Nadu',

 'Coimbatore': 'Tamil Nadu',

 'Delhi': 'Delhi',

 'Ernakulam': 'Kerala',

 'Gurugram': 'Haryana',

 'Guwahati': 'Assam',

 'Hyderabad': 'Andhra Pradesh',

 'Jaipur': 'Rajasthan',

 'Jorapokhar': 'Jharkhand',

 'Kochi': 'Kerala',

 'Kolkata': 'West Bengal',

 'Lucknow': 'Uttar Pradesh',

 'Mumbai': 'Maharashtra',

 'Patna': 'Bihar',

 'Shillong': 'Meghalaya',

 'Talcher': 'Orissa',

 'Thiruvananthapuram': 'Kerala',

 'Visakhapatnam': 'Andhra Pradesh'}



locations = {'Adarsh Nagar, Jaipur - RSPCB': [26.8986698, 75.8163567],

 'Alandur Bus Depot, Chennai - CPCB': [12.9938386, 80.1622219],

 'Alipur, Delhi - DPCC': [28.7959955, 77.1360706],

 'Anand Vihar, Delhi - DPCC': [28.6256914, 77.10194106560284],

 'Ashok Vihar, Delhi - DPCC': [28.6994533, 77.1848256],

 'Aya Nagar, Delhi - IMD': [28.47649105, 77.13291315925144],

 'BTM Layout, Bengaluru - CPCB': [12.9151772, 77.6102821],

 'BWSSB Kadabesanahalli, Bengaluru - CPCB': [12.9791198, 77.5912997],

 'Ballygunge, Kolkata - WBPCB': [22.5258813, 88.3660468],

 'Bandra, Mumbai - MPCB': [19.0549792, 72.8402203],

 'Bapuji Nagar, Bengaluru - KSPCB': [12.9556699, 77.5402492],

 'Bawana, Delhi - DPCC': [28.79966, 77.0328847],

 'Bidhannagar, Kolkata - WBPCB': [22.58162035, 88.4523869411636],

 'Bollaram Industrial Area, Hyderabad - TSPCB': [17.5026977, 78.3645689],

 'Borivali East, Mumbai - MPCB': [19.2267228, 72.8619328],

 'Burari Crossing, Delhi - IMD': [28.7285944, 77.1993251],

 'CRRI Mathura Road, Delhi - IMD': [28.5500925, 77.2751557],

 'Central School, Lucknow - CPCB': [26.8528761, 77.79518485076366],

 'Central University, Hyderabad - TSPCB': [17.4031425, 78.45593431157263],

 'Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB': [18.9387711,

  72.8353355],

 'City Railway Station, Bengaluru - KSPCB': [12.9791198, 77.5912997],

 'Colaba, Mumbai - MPCB': [18.915091, 72.8259691],

 'DRM Office Danapur, Patna - BSPCB': [25.6093239, 85.1235252],

 'DTU, Delhi - CPCB': [28.7473272, 77.11568781182513],

 'Dr. Karni Singh Shooting Range, Delhi - DPCC': [28.5000822,

  77.26751472410736],

 'Dwarka-Sector 8, Delhi - DPCC': [28.5656109, 77.0670366],

 'East Arjun Nagar, Delhi - CPCB': [28.6569534, 77.2947178],

 'Fort William, Kolkata - WBPCB': [22.55449655, 88.33801318408831],

 'GM Office, Brajrajnagar - OSPCB': [20.296059, 85.824539],

 'GVM Corporation, Visakhapatnam - APPCB': [17.7231276, 83.3012842],

 'Golden Temple, Amritsar - PPCB': [31.61997565, 74.87654076032885],

 'Gomti Nagar, Lucknow - UPPCB': [26.8528761, 80.9988505],

 'Govt. High School Shikarpur, Patna - BSPCB': [25.6093239, 85.1235252],

 'Hebbal, Bengaluru - KSPCB': [13.0382184, 77.5919],

 'Hombegowda Nagar, Bengaluru - KSPCB': [12.9791198, 77.5912997],

 'ICRISAT Patancheru, Hyderabad - TSPCB': [17.38878595, 78.46106473453146],

 'IDA Pashamylaram, Hyderabad - TSPCB': [17.38878595, 78.46106473453146],

 'IGI Airport (T3), Delhi - IMD': [28.55489735, 77.08467458266915],

 'IGSC Planetarium Complex, Patna - BSPCB': [25.6093239, 85.1235252],

 'IHBAS, Dilshad Garden, Delhi - CPCB': [28.6517178, 77.2219388],

 'ITO, Delhi - CPCB': [28.6305091, 77.2414363],

 'Jadavpur, Kolkata - WBPCB': [22.4951079, 88.3749813],

 'Jahangirpuri, Delhi - DPCC': [28.7259717, 77.162658],

 'Jawaharlal Nehru Stadium, Delhi - DPCC': [28.58337705, 77.23354040287734],

 'Jayanagar 5th Block, Bengaluru - KSPCB': [12.9194605, 77.58332110317872],

 'Kacheripady, Ernakulam - Kerala PCB': [9.9862208, 76.2831865],

 'Kariavattom, Thiruvananthapuram - Kerala PCB': [8.5614821, 76.8829796],

 'Kurla, Mumbai - MPCB': [19.0652797, 72.8793805],

 'Lalbagh, Lucknow - CPCB': [26.7908341, 80.8701804],

 'Lodhi Road, Delhi - IMD': [28.589461, 77.2128399],

 'Lumpyngngad, Shillong - Meghalaya PCB': [25.5760446, 91.8825282],

 'Major Dhyan Chand National Stadium, Delhi - DPCC': [28.612634200000002,

  77.23733038046025],

 'Manali Village, Chennai - TNPCB': [13.1672275, 80.2598107],

 'Manali, Chennai - CPCB': [13.1672275, 80.2598107],

 'Mandir Marg, Delhi - DPCC': [28.5246601, 77.216217],

 'Maninagar, Ahmedabad - GPCB': [22.9977135, 72.6067174],

 'Mundka, Delhi - DPCC': [28.6824341, 77.0305741],

 'Muradpur, Patna - BSPCB': [28.1128875, 75.8935118],

 'NISE Gwal Pahari, Gurugram - IMD': [28.4646148, 77.0299194],

 'NSIT Dwarka, Delhi - CPCB': [28.6082819, 77.0350079],

 'Najafgarh, Delhi - DPCC': [28.612304, 76.9823908],

 'Narela, Delhi - DPCC': [28.8426096, 77.0918354],

 'Nehru Nagar, Delhi - DPCC': [28.5685108, 77.2513847],

 'Nishant Ganj, Lucknow - UPPCB': [26.8381, 80.9346001],

 'North Campus, DU, Delhi - IMD': [32.8880156, -117.24108559234455],

 'Okhla Phase-2, Delhi - DPCC': [28.5366138, 77.2756197],

 'Patparganj, Delhi - DPCC': [28.6115923, 77.2905644],

 'Peenya, Bengaluru - CPCB': [13.0329419, 77.5273253],

 'Plammoodu, Thiruvananthapuram - Kerala PCB': [8.5241122, 76.9360573],

 'Police Commissionerate, Jaipur - RSPCB': [26.916194, 75.820349],

 'Powai, Mumbai - MPCB': [19.1187195, 72.9073476],

 'Punjabi Bagh, Delhi - DPCC': [28.668945, 77.1324614],

 'Pusa, Delhi - DPCC': [28.641230399999998, 77.1742940078465],

 'Pusa, Delhi - IMD': [28.641230399999998, 77.1742940078465],

 'R K Puram, Delhi - DPCC': [28.5503864, 77.1855171],

 'Rabindra Bharati University, Kolkata - WBPCB': [22.62696605,

  88.38049809655342],

 'Rabindra Sarobar, Kolkata - WBPCB': [22.51226105, 88.36383105710377],

 'Railway Colony, Guwahati - APCB': [26.1808827, 91.7824864],

 'Rajbansi Nagar, Patna - BSPCB': [25.6093239, 85.1235252],

 'Rohini, Delhi - DPCC': [28.7162092, 77.1170743],

 'SIDCO Kurichi, Coimbatore - TNPCB': [11.0018115, 76.9628425],

 'Samanpura, Patna - BSPCB': [25.6093239, 85.1235252],

 'Sanathnagar, Hyderabad - TSPCB': [17.4569654, 78.4434780636594],

 'Sanegurava Halli, Bengaluru - KSPCB': [12.9791198, 77.5912997],

 'Secretariat, Amaravati - APPCB': [16.5134691, 80.517227],

 'Sector-25, Chandigarh - CPCC': [30.7516466, 76.7567324],

 'Sector-51, Gurugram - HSPCB': [28.4287011, 77.0666877],

 'Shadipur, Delhi - CPCB': [28.6516362, 77.1582947],

 'Shastri Nagar, Jaipur - RSPCB': [25.787581, -100.4685005],

 'Sikulpuikawn, Aizawl - Mizoram PCB': [23.7414092, 92.7209297],

 'Silk Board, Bengaluru - KSPCB': [12.9167139, 77.6214094],

 'Sion, Mumbai - MPCB': [19.0465213, 72.8632834],

 'Sirifort, Delhi - CPCB': [28.6517178, 77.2219388],

 'Sonia Vihar, Delhi - DPCC': [28.7199257, 77.2481823],

 'Sri Aurobindo Marg, Delhi - DPCC': [28.5396291, 77.2000301],

 'T T Nagar, Bhopal - MPPCB': [23.2286993, 77.4002881],

 'Talcher Coalfields,Talcher - OSPCB': [20.9458183, 85.2111736],

 'Talkatora District Industries Center, Lucknow - CPCB': [26.8381, 80.9346001],

 'Tata Stadium, Jorapokhar - JSPCB': [23.7167069, 86.4110166],

 'Teri Gram, Gurugram - HSPCB': [28.35117845, 77.06446614772139],

 'Vasai West, Mumbai - MPCB': [19.3849292, 72.897546],

 'Velachery Res. Area, Chennai - CPCB': [12.980165450000001,

  80.22285056225584],

 'Victoria, Kolkata - WBPCB': [22.54978375, 88.33911363126559],

 'Vikas Sadan, Gurugram - HSPCB': [28.4646148, 77.0299194],

 'Vile Parle West, Mumbai - MPCB': [19.1038725, 72.8402903],

 'Vivek Vihar, Delhi - DPCC': [28.6691641, 77.31226695421603],

 'Vyttila, Kochi - Kerala PCB': [9.9701655, 76.3180562],

 'Wazirpur, Delhi - DPCC': [28.680084299999997, 77.17022123990277],

 'Worli, Mumbai - MPCB': [19.0116962, 72.8180702],

 'Zoo Park, Hyderabad - TSPCB': [25.3841041, 68.3413739]}





street_address = {'Adarsh Nagar, Jaipur - RSPCB': 'Adarsh Nagar, Jaipur', 'Alandur Bus Depot, Chennai - CPCB': 'Alandur Bus Depot, Chennai', 'Alipur, Delhi - DPCC': 'Alipur, Delhi', 'Anand Vihar, Delhi - DPCC': 'Anand Vihar, Delhi', 'Ashok Vihar, Delhi - DPCC': 'Ashok Vihar, Delhi', 'Aya Nagar, Delhi - IMD': 'Aya Nagar, Delhi', 'BTM Layout, Bengaluru - CPCB': 'BTM Layout, Bengaluru', 'BWSSB Kadabesanahalli, Bengaluru - CPCB': 'BWSSB Kadabesanahalli, Bengaluru', 'Ballygunge, Kolkata - WBPCB': 'Ballygunge, Kolkata', 'Bandra, Mumbai - MPCB': 'Bandra, Mumbai', 'Bapuji Nagar, Bengaluru - KSPCB': 'Bapuji Nagar, Bengaluru', 'Bawana, Delhi - DPCC': 'Bawana, Delhi', 'Bidhannagar, Kolkata - WBPCB': 'Bidhannagar, Kolkata', 'Bollaram Industrial Area, Hyderabad - TSPCB': 'Bollaram Industrial Area, Hyderabad', 'Borivali East, Mumbai - MPCB': 'Borivali East, Mumbai', 'Burari Crossing, Delhi - IMD': 'Burari Crossing, Delhi', 'CRRI Mathura Road, Delhi - IMD': 'CRRI Mathura Road, Delhi', 'Central School, Lucknow - CPCB': 'Central School, Lucknow', 'Central University, Hyderabad - TSPCB': 'Central University, Hyderabad', 'Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB': 'Chhatrapati Shivaji Intl. Airport (T2), Mumbai', 'City Railway Station, Bengaluru - KSPCB': 'City Railway Station, Bengaluru', 'Colaba, Mumbai - MPCB': 'Colaba, Mumbai', 'DRM Office Danapur, Patna - BSPCB': 'DRM Office Danapur, Patna', 'DTU, Delhi - CPCB': 'DTU, Delhi', 'Dr. Karni Singh Shooting Range, Delhi - DPCC': 'Dr. Karni Singh Shooting Range, Delhi', 'Dwarka-Sector 8, Delhi - DPCC': 'Dwarka-Sector 8, Delhi', 'East Arjun Nagar, Delhi - CPCB': 'East Arjun Nagar, Delhi', 'Fort William, Kolkata - WBPCB': 'Fort William, Kolkata', 'GM Office, Brajrajnagar - OSPCB': 'GM Office, Brajrajnagar', 'GVM Corporation, Visakhapatnam - APPCB': 'GVM Corporation, Visakhapatnam', 'Golden Temple, Amritsar - PPCB': 'Golden Temple, Amritsar', 'Gomti Nagar, Lucknow - UPPCB': 'Gomti Nagar, Lucknow', 'Govt. High School Shikarpur, Patna - BSPCB': 'Govt. High School Shikarpur, Patna', 'Hebbal, Bengaluru - KSPCB': 'Hebbal, Bengaluru', 'Hombegowda Nagar, Bengaluru - KSPCB': 'Hombegowda Nagar, Bengaluru', 'ICRISAT Patancheru, Hyderabad - TSPCB': 'ICRISAT Patancheru, Hyderabad', 'IDA Pashamylaram, Hyderabad - TSPCB': 'IDA Pashamylaram, Hyderabad', 'IGI Airport (T3), Delhi - IMD': 'IGI Airport (T3), Delhi', 'IGSC Planetarium Complex, Patna - BSPCB': 'IGSC Planetarium Complex, Patna', 'IHBAS, Dilshad Garden, Delhi - CPCB': 'IHBAS, Dilshad', 'ITO, Delhi - CPCB': 'ITO, Delhi', 'Jadavpur, Kolkata - WBPCB': 'Jadavpur, Kolkata', 'Jahangirpuri, Delhi - DPCC': 'Jahangirpuri, Delhi', 'Jawaharlal Nehru Stadium, Delhi - DPCC': 'Jawaharlal Nehru Stadium, Delhi', 'Jayanagar 5th Block, Bengaluru - KSPCB': 'Jayanagar 5th Block, Bengaluru', 'Kacheripady, Ernakulam - Kerala PCB': 'Kacheripady, Ernakulam', 'Kariavattom, Thiruvananthapuram - Kerala PCB': 'Kariavattom, Thiruvananthapuram', 'Kurla, Mumbai - MPCB': 'Kurla, Mumbai', 'Lalbagh, Lucknow - CPCB': 'Lalbagh, Lucknow', 'Lodhi Road, Delhi - IMD': 'Lodhi Road, Delhi', 'Lumpyngngad, Shillong - Meghalaya PCB': 'Lumpyngngad, Shillong', 'Major Dhyan Chand National Stadium, Delhi - DPCC': 'Major Dhyan Chand National Stadium, Delhi', 'Manali Village, Chennai - TNPCB': 'Manali Village, Chennai', 'Manali, Chennai - CPCB': 'Manali, Chennai', 'Mandir Marg, Delhi - DPCC': 'Mandir Marg, Delhi', 'Maninagar, Ahmedabad - GPCB': 'Maninagar, Ahmedabad', 'Mundka, Delhi - DPCC': 'Mundka, Delhi', 'Muradpur, Patna - BSPCB': 'Muradpur, Patna', 'NISE Gwal Pahari, Gurugram - IMD': 'NISE Gwal Pahari, Gurugram', 'NSIT Dwarka, Delhi - CPCB': 'NSIT Dwarka, Delhi', 'Najafgarh, Delhi - DPCC': 'Najafgarh, Delhi', 'Narela, Delhi - DPCC': 'Narela, Delhi', 'Nehru Nagar, Delhi - DPCC': 'Nehru Nagar, Delhi', 'Nishant Ganj, Lucknow - UPPCB': 'Nishant Ganj, Lucknow', 'North Campus, DU, Delhi - IMD': 'North Campus, DU,', 'Okhla Phase-2, Delhi - DPCC': 'Okhla Phase-2, Delhi', 'Patparganj, Delhi - DPCC': 'Patparganj, Delhi', 'Peenya, Bengaluru - CPCB': 'Peenya, Bengaluru', 'Plammoodu, Thiruvananthapuram - Kerala PCB': 'Plammoodu, Thiruvananthapuram', 'Police Commissionerate, Jaipur - RSPCB': 'Police Commissionerate, Jaipur', 'Powai, Mumbai - MPCB': 'Powai, Mumbai', 'Punjabi Bagh, Delhi - DPCC': 'Punjabi Bagh, Delhi', 'Pusa, Delhi - DPCC': 'Pusa, Delhi', 'Pusa, Delhi - IMD': 'Pusa, Delhi', 'R K Puram, Delhi - DPCC': 'R K Puram, Delhi', 'Rabindra Bharati University, Kolkata - WBPCB': 'Rabindra Bharati University, Kolkata', 'Rabindra Sarobar, Kolkata - WBPCB': 'Rabindra Sarobar, Kolkata', 'Railway Colony, Guwahati - APCB': 'Railway Colony, Guwahati', 'Rajbansi Nagar, Patna - BSPCB': 'Rajbansi Nagar, Patna', 'Rohini, Delhi - DPCC': 'Rohini, Delhi', 'SIDCO Kurichi, Coimbatore - TNPCB': 'SIDCO Kurichi, Coimbatore', 'Samanpura, Patna - BSPCB': 'Samanpura, Patna', 'Sanathnagar, Hyderabad - TSPCB': 'Sanathnagar, Hyderabad', 'Sanegurava Halli, Bengaluru - KSPCB': 'Sanegurava Halli, Bengaluru', 'Secretariat, Amaravati - APPCB': 'Secretariat, Amaravati', 'Sector-25, Chandigarh - CPCC': 'Sector-25, Chandigarh', 'Sector-51, Gurugram - HSPCB': 'Sector-51, Gurugram', 'Shadipur, Delhi - CPCB': 'Shadipur, Delhi', 'Shastri Nagar, Jaipur - RSPCB': 'Shastri Nagar, Jaipur', 'Sikulpuikawn, Aizawl - Mizoram PCB': 'Sikulpuikawn, Aizawl', 'Silk Board, Bengaluru - KSPCB': 'Silk Board, Bengaluru', 'Sion, Mumbai - MPCB': 'Sion, Mumbai', 'Sirifort, Delhi - CPCB': 'Sirifort, Delhi', 'Sonia Vihar, Delhi - DPCC': 'Sonia Vihar, Delhi', 'Sri Aurobindo Marg, Delhi - DPCC': 'Sri Aurobindo Marg, Delhi', 'T T Nagar, Bhopal - MPPCB': 'T T Nagar, Bhopal', 'Talcher Coalfields,Talcher - OSPCB': 'Talcher Coalfields,Talcher - OSPCB ', 'Talkatora District Industries Center, Lucknow - CPCB': 'Talkatora District Industries Center, Lucknow', 'Tata Stadium, Jorapokhar - JSPCB': 'Tata Stadium, Jorapokhar', 'Teri Gram, Gurugram - HSPCB': 'Teri Gram, Gurugram', 'Vasai West, Mumbai - MPCB': 'Vasai West, Mumbai', 'Velachery Res. Area, Chennai - CPCB': 'Velachery Res. Area, Chennai', 'Victoria, Kolkata - WBPCB': 'Victoria, Kolkata', 'Vikas Sadan, Gurugram - HSPCB': 'Vikas Sadan, Gurugram', 'Vile Parle West, Mumbai - MPCB': 'Vile Parle West, Mumbai', 'Vivek Vihar, Delhi - DPCC': 'Vivek Vihar, Delhi', 'Vyttila, Kochi - Kerala PCB': 'Vyttila, Kochi', 'Wazirpur, Delhi - DPCC': 'Wazirpur, Delhi', 'Worli, Mumbai - MPCB': 'Worli, Mumbai', 'Zoo Park, Hyderabad - TSPCB': 'Zoo Park, Hyderabad'}
#adding states and state ID to df_city



df_city['State'] = df_city['City'].apply(lambda x: city_states[x])

df_city['State_ID'] = df_city['State'].apply(lambda x: state_id_map[x])



df_stations['street address'] = df_stations['StationName'].apply(lambda x: street_address[x])

df_stations['latitude'] = df_stations['StationName'].apply(lambda x: locations[x][0])

df_stations['longitude'] = df_stations['StationName'].apply(lambda x: locations[x][1])



df_stations['AQI_Bucket'] = df_stations['AQI'].apply(get_AQI_bucket)
# locations = {}

# street_address ={}

# stations = df_stations.groupby('StationName').City.max()



# for i in range(0,len(stations)):

#     print(i)

#     locations[stations.index[i]] = get_geo(stations.index[i],stations[i])

#     street_address[stations.index[i]] = get_address(stations.index[i])

    

# # print(locations)

# print(street_address)
map_data = df_city[['Date','City','AQI','State','State_ID']]

map_data['Date'] = pd.to_datetime(map_data['Date'])



map_data = map_data.set_index('Date')

map_data = map_data.loc['2019-11-07']

map_data.reset_index(drop= False,inplace = True)

map_data['Date'] = map_data['Date'].apply(lambda x: x.strftime('%m/%d/%y'))
indiamap = px.choropleth(data_frame = map_data,locations = 'State_ID', geojson = india_states,

                         color = 'AQI',hover_name = 'State',hover_data = {'AQI':True,'State_ID':False,'Date':True},

                         color_continuous_scale=px.colors.sequential.RdBu_r)

indiamap.update_geos(fitbounds = 'locations',visible = False)



# indiamap.update_layout(

#     title_text = 'State-wise AQI',

#     title_x = 0.5,

#     geo=dict(

#         showframe = False,

#         showcoastlines = False,

#     ))



indiamap.show()
city_aqi_data = df_city[['City','AQI',"Date"]]



latest_station_report = df_stations[df_stations['Date'] == '2020-07-01'].reset_index(drop = True)



india_map = folium.Map(location = [21, 78],zoom_start = 5.4,tiles =  'CartoDB positron',max_zoom = 15,min_zoom = 5)

color_dict = {'Satisfactory':'Green', 'Good':"light blue", 'Moderate':"Orange", 'Not Provided' :"white", 'Poor':"Red",'Very Poor':'Maroon','Severe':'Purple'}



for i in range(0,len(latest_station_report)):

    folium.Circle(location  = [latest_station_report['latitude'][i],latest_station_report['longitude'][i]],

                  tooltip ="<h5 style = 'text-align:center; font-weight:bold'>" +"Station Name:"+"</h5>"+ str(latest_station_report['street address'][i])+\

                  "<h5 style = 'text-align:center; font-weight:bold'>"+ '\n\n AQI:' + str(latest_station_report['AQI'][i]),

                 radius = 10000, color = color_dict[latest_station_report['AQI_Bucket'][i]],

                 fill_color = color_dict[latest_station_report['AQI_Bucket'][i]],fill = True).add_to(india_map)

    

india_map
city_mean_aqi = df_city.groupby('City')['AQI'].mean()

city_mean_aqi  = pd.DataFrame(city_mean_aqi)

city_mean_aqi['AQI_Bucket'] = city_mean_aqi['AQI'].apply(get_AQI_bucket)

city_mean_aqi['AQI'] = round(city_mean_aqi['AQI'],2)

city_mean_aqi.reset_index(drop = False, inplace = True)



# d = dict(city_data[['City','AQI_Bucket']])



state_aqi_buckets ={'Ahmedabad': 'Severe',

 'Aizawl': 'Good',

 'Amaravati': 'Satisfactory',

 'Amritsar': 'Moderate',

 'Bengaluru': 'Satisfactory',

 'Bhopal': 'Moderate',

 'Brajrajnagar': 'Moderate',

 'Chandigarh': 'Satisfactory',

 'Chennai': 'Moderate',

 'Coimbatore': 'Satisfactory',

 'Delhi': 'Poor',

 'Ernakulam': 'Satisfactory',

 'Gurugram': 'Moderate',

 'Guwahati': 'Moderate',

 'Hyderabad': 'Moderate',

 'Jaipur': 'Moderate',

 'Jorapokhar': 'Moderate',

 'Kochi': 'Moderate',

 'Kolkata': 'Moderate',

 'Lucknow': 'Poor',

 'Mumbai': 'Good',

 'Patna': 'Moderate',

 'Shillong': 'Good',

 'Talcher': 'Moderate',

 'Thiruvananthapuram': 'Satisfactory',

 'Visakhapatnam': 'Satisfactory'}

city_aqi_data = df_city[['City','AQI',"Date"]]

city_aqi_data['Average AQI Bucket'] = city_aqi_data['City'].apply(lambda x: state_aqi_buckets[x])





fig1 = px.box(data_frame= city_aqi_data, x = 'City', y = 'AQI',

       template = 'ggplot2',color = 'Average AQI Bucket', color_discrete_sequence= ["black", "green", "orange", "blue","red"],

       hover_name = 'AQI',hover_data = {'AQI':False,'Date':True,'City':False},title = "State-wise AQI Spread (from 2015 to Present) ",

       labels = {'City':""})



fig1.update_layout(xaxis={'categoryorder':'category ascending'})



# X-----------------------------------------------------------X



fig2 = px.sunburst(data_frame= city_mean_aqi , path = city_mean_aqi[['AQI_Bucket','City']],template = 'ggplot2',

                   color = city_mean_aqi['AQI_Bucket'], color_discrete_sequence = ["lightgreen", "lightblue", "red", "yellow","black"],

                   hover_data = {'AQI_Bucket':False,'AQI': True}, hover_name = 'AQI', title = 'Average AQI (between 2015 and present) and AQI buckets',

                  labels = {'AQI':"Average AQI"})



fig2.update_layout(title = {'text':'State-wise Average AQI (between 2015 and Present)','y':0.95,'x':0.5,'xanchor':'center'})





# X-------------------------------------------------------------X



fig1.show()

fig2.show()
all_india = df_city.groupby('Date')



mean_national_aqi = all_india['AQI'].mean().reset_index()

mean_national_aqi_bucket=[]

mean_national_aqi['AQI'] = round(mean_national_aqi['AQI'],2)



dates =  pd.to_datetime(mean_national_aqi['Date'].values)

mean_national_aqi['Date'] = dates.strftime("%m/%d")

years = dates.year

mean_national_aqi['year'] = years



# months = dates.month

# days =dates.day



for i in mean_national_aqi['AQI']:

   mean_national_aqi_bucket.append(get_AQI_bucket(i))



# mean_national_aqi['year'] = years







fig= px.bar(data_frame =mean_national_aqi,x = mean_national_aqi['Date'],

           y = (mean_national_aqi['AQI']),

           template = 'ggplot2',color = mean_national_aqi_bucket,

           color_discrete_sequence = ["blue", "red", "orange", "maroon","black"],

           title = "India's Quality of Air since 2015",hover_name = mean_national_aqi['AQI'],

           labels = {'color':'AQI_Bucket ','value':'AQI'},facet_row = 'year',height = 1500)



fig.update_layout(xaxis={'categoryorder':'category ascending'})





fig.update_yaxes(matches=None)

fig.update_xaxes(matches ='x')

fig.show()
dates = pd.to_datetime(df_city['Date'].values)

df_city['month'] = dates.month

df_city['month_name'] = dates.month_name()



aqi_month = df_city.groupby(['month','month_name'])['AQI'].mean().reset_index()

aqi_month['AQI'] = round(aqi_month['AQI'],2)

average = aqi_month['AQI'].mean()



px.bar(data_frame= aqi_month, x = aqi_month['month_name'], y = (aqi_month['AQI'].values - average) ,color = 'AQI',

       color_continuous_scale=px.colors.sequential.Bluered,title = 'The Average AQI of India: ' + str(round(average,2)) +" (from 2015 to present)",

       template = 'ggplot2',labels = {'y':'Monthly Variance in AQI from Average ','month_name':''},hover_data={'month_name':False},width = 750,height = 500)
df_city['AQI_Bucket'] = df_city['AQI'].apply(lambda x: get_AQI_bucket(x))

delhi =  df_city[df_city['City'] == 'Delhi']



dates = pd.to_datetime(delhi['Date'].values)

delhi['Date'] = dates.strftime("%m/%d")

years = dates.year

delhi['year'] = years





delhi.set_index('year',inplace = True)

delhi = delhi.loc['2015':'2019']

delhi.reset_index(inplace = True,)









fig= px.bar(data_frame= delhi, x ='Date' ,y = 'AQI', template = 'ggplot2',color = delhi['AQI_Bucket'],

       color_discrete_sequence =["Black", "Blue", "Maroon", "Red","yellow",'lightgreen'],

           title = "Delhi's Quality of Air since 2015",hover_name = 'AQI_Bucket',facet_row = 'year',height = 1500)



fig.update_layout(xaxis={'categoryorder':'category ascending'})



fig.show()
delhi =  df_city[df_city['City'] == 'Delhi']



dates = pd.to_datetime(delhi['Date'].values)

delhi['month'] = dates.month

delhi['month_name'] = dates.month_name()



aqi_month = delhi.groupby(['month','month_name'])['AQI'].mean().reset_index()

aqi_month['AQI'] = round(aqi_month['AQI'],2)

average = round(aqi_month['AQI'].mean(),2)



px.bar(data_frame= aqi_month, x = aqi_month['month_name'], y = (aqi_month['AQI'].values - average) ,color = 'AQI',

       color_continuous_scale=px.colors.sequential.Bluered,template = 'ggplot2',

      labels = {'y':'Monthly Variance in AQI from the Average','month_name':''},hover_name = aqi_month['month_name'],

      hover_data={'month_name':False},title = ' Average AQI of Delhi: '+str(average),width = 750,height = 500)
from fbprophet import Prophet 



delhi_aqi = delhi[['Date','AQI']]

delhi_aqi.reset_index(inplace = True,drop = True)



train_df = delhi_aqi

train_df.rename(mapper = {'Date':'ds','AQI':'y'},axis =1,inplace = True)

train_df



m = Prophet(holidays_prior_scale=0,seasonality_prior_scale=20,n_changepoints= 50,)

m.fit(train_df)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from fbprophet.diagnostics import mape,cross_validation,performance_metrics

df_cv = cross_validation(m, initial='1100 days', period='121 days', horizon = '365 days')

df_p = performance_metrics(df_cv)

print('Cross Validation accuracy:', (1 - df_p['mape'].mean())*100)
from fbprophet.plot import plot_plotly, plot_components_plotly



fig = plot_plotly(m, forecast ,xlabel = 'Date',ylabel= 'AQI',figsize=(1000,750))



fig.show()