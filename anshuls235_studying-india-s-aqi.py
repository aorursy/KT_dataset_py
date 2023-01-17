!pip install googlemaps
import numpy as np

import pandas as pd

import googlemaps

import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import datetime as dt

import warnings

warnings.filterwarnings("ignore")
#Values used throughout the kernel

colorscale = ['#77DD77','#33AF13','#F6D20E','#F17700','#FE6B64','#F12424']

PAPER_BGCOLOR = '#f5f2d0'

BGCOLOR = 'LightSteelBlue'
fig = go.Figure(data=[go.Table(

    columnorder = [1,2,3,4],

    columnwidth = [50,70,60,400],

    

    header=dict(values=['<b>AQI</b>', '<b>Remark</b>','<b>Colour Code</b>','<b>Possible Health Effects</b>'],

                line_color='darkslategray',

                fill_color='skyblue',

                align='left'),

    cells=dict(values=[['0-50','51-100','101-200','201-300','301-400','401-500'],

                       ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe'],

                       ['','','','','',''],

                       ['Minimal impact','Minor breathing discomfort to sensitive people',\

                       'Breathing discomfort to the people with lungs, asthma and heart diseases',\

                       'Breathing discomfort to most people on prolonged exposure',\

                       'Respiratory illness on prolonged exposure','Affects healthy people and seriously impacts those with existing diseases']],

               line_color='darkslategray',

               fill_color=['rgb(255,255,255)',

                           'rgb(255,255,255)',

                            [color for color in colorscale],

                           'rgb(255,255,255)'],

               align='left'))

])



fig.update_layout(height=180,paper_bgcolor='LightSteelBlue',margin=dict(l=5,r=5,t=5,b=5))

fig.show()
df_cd = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')

df_ch = pd.read_csv('/kaggle/input/air-quality-data-in-india/city_hour.csv')

df_sd = pd.read_csv('/kaggle/input/air-quality-data-in-india/station_day.csv')

df_sh = pd.read_csv('/kaggle/input/air-quality-data-in-india/station_hour.csv')

df_st = pd.read_csv('/kaggle/input/air-quality-data-in-india/stations.csv')

df_temp = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("gmaps")

secret_value_1 = user_secrets.get_secret("mapboxtoken")

gmaps = googlemaps.Client(key=secret_value_0)



#This is the logic used to fetch the coordibates of stations

# d = {}

# stations = df_st.StationName.unique().tolist()

# for station in stations:

#     add = station.split('-')[0].strip()

#     data = gmaps.geocode(add)

#     d[station] = data[0]['geometry']['location']



def get_coords(state):

    lat = d[state]['lat']

    lng = d[state]['lng']

    return (lat,lng)

#This dictionary is derived using above commented code

d = { 'Adarsh Nagar, Jaipur - RSPCB': {'lat': 26.9018769, 'lng': 75.8271749},

    'Airoli, Navi Mumbai - MPCB': {'lat': 19.159014, 'lng': 72.9985686},

    'Alandur Bus Depot, Chennai - CPCB': {   'lat': 12.9956231,

                                             'lng': 80.18900219999999},

    'Alipur, Delhi - DPCC': {'lat': 28.7972263, 'lng': 77.13313629999999},

    'Anand Kala Kshetram, Rajamahendravaram - APPCB': {   'lat': 17.0082015,

                                                          'lng': 81.77145949999999},

    'Anand Vihar, Delhi - DPCC': {'lat': 28.650218, 'lng': 77.30270589999999},

    'Anand Vihar, Hapur - UPPCB': {'lat': 28.7222506, 'lng': 77.7537913},

    'Ardhali Bazar, Varanasi - UPPCB': {   'lat': 25.3476918,

                                           'lng': 82.98088609999999},

    'Arya Nagar, Bahadurgarh - HSPCB': {'lat': 28.6771177, 'lng': 76.9282948},

    'Asansol Court Area, Asansol - WBPCB': {   'lat': 23.6871984,

                                               'lng': 86.9461361},

    'Ashok Nagar, Udaipur - RSPCB': {   'lat': 24.5886328,

                                        'lng': 73.70218009999999},

    'Ashok Vihar, Delhi - DPCC': {'lat': 28.6909791, 'lng': 77.17652389999999},

    'Aya Nagar, Delhi - IMD': {'lat': 28.4720443, 'lng': 77.1329417},

    'BTM Layout, Bengaluru - CPCB': {'lat': 12.9165757, 'lng': 77.6101163},

    'BWSSB Kadabesanahalli, Bengaluru - CPCB': {   'lat': 12.9603881,

                                                   'lng': 77.71899309999999},

    'Ballygunge, Kolkata - WBPCB': {   'lat': 22.5280352,

                                       'lng': 88.36590830000002},

    'Bandhavgar Colony, Satna - Birla Cement': {   'lat': 24.5798734,

                                                   'lng': 80.8557684},

    'Bandra, Mumbai - MPCB': {'lat': 19.0595596, 'lng': 72.8295287},

    'Bapuji Nagar, Bengaluru - KSPCB': {   'lat': 12.95678,

                                           'lng': 77.53972929999999},

    'Bawana, Delhi - DPCC': {'lat': 28.793229, 'lng': 77.0483355},

    'Belur Math, Howrah - WBPCB': {'lat': 22.6280889, 'lng': 88.35176380000001},

    'Bhopal Chauraha, Dewas - MPPCB': {'lat': 22.9687824, 'lng': 76.0635763},

    'Bidhannagar, Kolkata - WBPCB': {'lat': 22.5796842, 'lng': 88.414312},

    'Bollaram Industrial Area, Hyderabad - TSPCB': {   'lat': 17.5432727,

                                                       'lng': 78.3514088},

    'Borivali East, Mumbai - MPCB': {   'lat': 19.2297814,

                                        'lng': 72.86085589999999},

    'Burari Crossing, Delhi - IMD': {'lat': 28.72852, 'lng': 77.199327},

    'CRRI Mathura Road, Delhi - IMD': {'lat': 28.5517202, 'lng': 77.2750344},

    'Central School, Lucknow - CPCB': {   'lat': 43.957268,

                                          'lng': -81.50915069999999},

    'Central University, Hyderabad - TSPCB': {   'lat': 17.4567372,

                                                 'lng': 78.32638399999999},

    'Chandrapur, Chandrapur - MPCB': {'lat': 19.9615398, 'lng': 79.2961468},

    'Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB': {   'lat': 19.0974373,

                                                                 'lng': 72.8745017},

    'Chhoti Gwaltoli, Indore - MPPCB': {'lat': 22.7152096, 'lng': 75.8700466},

    'Chikkaballapur Rural, Chikkaballapur - KSPCB': {   'lat': 13.4290654,

                                                        'lng': 77.73304739999999},

    'City Center, Gwalior - MPPCB': {'lat': 26.2035227, 'lng': 78.1920664},

    'City Railway Station, Bengaluru - KSPCB': {   'lat': 12.9781291,

                                                   'lng': 77.5695295},

    'Civil Line, Jalandhar - PPCB': {'lat': 31.3208006, 'lng': 75.5793401},

    'Civil Lines, Ajmer - RSPCB': {'lat': 26.4726871, 'lng': 74.6415071},

    'Colaba, Mumbai - MPCB': {'lat': 18.9067031, 'lng': 72.8147123},

    'Collector Office, Yadgir - KSPCB': {   'lat': 16.751388,

                                            'lng': 77.13632969999999},

    'Collectorate, Gaya - BSPCB': {'lat': 24.7914917, 'lng': 85.006337},

    'Collectorate, Jodhpur - RSPCB': {'lat': 26.2918067, 'lng': 73.0366998},

    'DRM Office Danapur, Patna - BSPCB': {   'lat': 25.5856237,

                                             'lng': 85.04429929999999},

    'DTU, Delhi - CPCB': {'lat': 28.7451463, 'lng': 77.1169907},

    'Deen Dayal Nagar, Sagar - MPPCB': {'lat': 23.8641966, 'lng': 78.806407},

    'Deshpande Nagar, Hubballi - KSPCB': {'lat': 15.3547598, 'lng': 75.1384848},

    'Dr. Karni Singh Shooting Range, Delhi - DPCC': {   'lat': 28.4997268,

                                                        'lng': 77.2670954},

    'Dwarka-Sector 8, Delhi - DPCC': {   'lat': 28.570709,

                                         'lng': 77.072722},

    'East Arjun Nagar, Delhi - CPCB': {   'lat': 28.65617319999999,

                                          'lng': 77.29474669999999},

    'F-Block, Sirsa - HSPCB': {'lat': 37.032551, 'lng': -95.6242631},

    'Fort William, Kolkata - WBPCB': {'lat': 22.5542459, 'lng': 88.3358744},

    'GIDC, Ankleshwar - GPCB': {'lat': 21.6143446, 'lng': 73.01155969999999},

    'GIDC, Nandesari - Nandesari Ind. Association': {   'lat': 22.4089434,

                                                        'lng': 73.0962447},

    'GM Office, Brajrajnagar - OSPCB': {   'lat': 21.8546923,

                                           'lng': 83.92479949999999},

    'GVM Corporation, Visakhapatnam - APPCB': {   'lat': 17.6868159,

                                                  'lng': 83.2184815},

    'Ganga Nagar, Meerut - UPPCB': {'lat': 29.0009035, 'lng': 77.7599208},

    'Gangapur Road, Nashik - MPCB': {'lat': 20.0168226, 'lng': 73.735682},

    'General Hospital, Mandikhera - HSPCB': {   'lat': 27.9001526,

                                                'lng': 76.993775},

    'Ghusuri, Howrah - WBPCB': {'lat': 22.6114858, 'lng': 88.35401449999999},

    'Gobind Pura, Yamuna Nagar - HSPCB': {'lat': 30.1501492, 'lng': 77.2850239},

    'Golden Temple, Amritsar - PPCB': {   'lat': 31.61998029999999,

                                          'lng': 74.8764849},

    'Gole Bazar, Katni - MPPCB': {'lat': 23.8327424, 'lng': 80.3978186},

    'Gomti Nagar, Lucknow - UPPCB': {'lat': 26.8496217, 'lng': 81.0072193},

    'Govt. High School Shikarpur, Patna - BSPCB': {   'lat': 25.5931871,

                                                      'lng': 85.2272598},

    'H.B. Colony, Bhiwani - HSPCB': {'lat': 28.8082043, 'lng': 76.1361016},

    'Haldia, Haldia - WBPCB': {'lat': 22.0627164, 'lng': 88.0832934},

    'Hardev Nagar, Bathinda - PPCB': {'lat': 30.2420588, 'lng': 74.9175889},

    'Hebbal 1st Stage, Mysuru - KSPCB': {'lat': 12.3500813, 'lng': 76.6209903},

    'Hebbal, Bengaluru - KSPCB': {'lat': 13.0353557, 'lng': 77.59878739999999},

    'Hombegowda Nagar, Bengaluru - KSPCB': {   'lat': 12.9375448,

                                               'lng': 77.5948946},

    'Huda Sector, Fatehabad - HSPCB': {'lat': 29.5030525, 'lng': 75.4737974},

    'ICRISAT Patancheru, Hyderabad - TSPCB': {   'lat': 17.5110595,

                                                 'lng': 78.27519389999999},

    'IDA Pashamylaram, Hyderabad - TSPCB': {   'lat': 17.5324702,

                                               'lng': 78.1849427},

    'IGI Airport (T3), Delhi - IMD': {'lat': 28.5550838, 'lng': 77.0844015},

    'IGSC Planetarium Complex, Patna - BSPCB': {   'lat': 25.6107873,

                                                   'lng': 85.131507},

    'IHBAS, Dilshad Garden, Delhi - CPCB': {   'lat': 28.6811689,

                                               'lng': 77.3047121},

    'ITO, Delhi - CPCB': {'lat': 28.6293713, 'lng': 77.2413201},

    'Ibrahimpur, Vijayapura - KSPCB': {'lat': 16.8028639, 'lng': 75.726973},

    'Indira Colony Vistar, Pali - RSPCB': {   'lat': 25.7731026,

                                              'lng': 73.3502377},

    'Indirapuram, Ghaziabad - UPPCB': {'lat': 28.6460176, 'lng': 77.3695166},

    'Industrial Area, Hajipur - BSPCB': {   'lat': 25.6927811,

                                            'lng': 85.24011639999999},

    'Jadavpur, Kolkata - WBPCB': {'lat': 22.4954988, 'lng': 88.3709008},

    'Jahangirpuri, Delhi - DPCC': {'lat': 28.7296171, 'lng': 77.16663129999999},

    'Jai Bhim Nagar, Meerut - UPPCB': {   'lat': 28.9579131,

                                          'lng': 77.75951309999999},

    'Jawaharlal Nehru Stadium, Delhi - DPCC': {   'lat': 28.5828456,

                                                  'lng': 77.2343665},

    'Jayanagar 5th Block, Bengaluru - KSPCB': {   'lat': 12.920789,

                                                  'lng': 77.5841502},

    'Kacheripady, Ernakulam - Kerala PCB': {   'lat': 9.988280099999999,

                                               'lng': 76.28121949999999},

    'Kalal Majra, Khanna - PPCB': {'lat': 30.7406085, 'lng': 76.20523779999999},

    'Kalyana Nagara, Chikkamagaluru - KSPCB': {   'lat': 13.3230247,

                                                  'lng': 75.7967371},

    'Kariavattom, Thiruvananthapuram - Kerala PCB': {   'lat': 8.5678435,

                                                        'lng': 76.8908318},

    'Karve Road, Pune - MPCB': {'lat': 18.5033095, 'lng': 73.8197888},

    'Khadakpada, Kalyan - MPCB': {'lat': 19.2592249, 'lng': 73.12792689999999},

    'Knowledge Park - III, Greater Noida - UPPCB': {   'lat': 34.925234,

                                                       'lng': -81.0260196},

    'Knowledge Park - V, Greater Noida - UPPCB': {   'lat': 34.925234,

                                                     'lng': -81.0260196},

    'Kurla, Mumbai - MPCB': {'lat': 19.0726295, 'lng': 72.8844721},

    'Lajpat Nagar, Moradabad - UPPCB': {'lat': 28.8253591, 'lng': 78.7830383},

    'Lal Bahadur Shastri Nagar, Kalaburagi - KSPCB': {   'lat': 17.3203897,

                                                         'lng': 76.8194767},

    'Lalbagh, Lucknow - CPCB': {'lat': 26.8459624, 'lng': 80.9415089},

    'Lodhi Road, Delhi - IMD': {'lat': 28.5910626, 'lng': 77.2280791},

    'Loni, Ghaziabad - UPPCB': {'lat': 28.7333526, 'lng': 77.2986264},

    'Lumpyngngad, Shillong - Meghalaya PCB': {   'lat': 25.5585941,

                                                 'lng': 91.89848649999999},

    'MD University, Rohtak - HSPCB': {   'lat': 28.8768269,

                                         'lng': 76.62110799999999},

    'MIDC Khutala, Chandrapur - MPCB': {   'lat': 19.9756764,

                                           'lng': 79.24229559999999},

    'Mahakaleshwar Temple, Ujjain - MPPCB': {   'lat': 23.1827177,

                                                'lng': 75.7682178},

    'Mahape, Navi Mumbai - MPCB': {'lat': 19.1182937, 'lng': 73.0275875},

    'Major Dhyan Chand National Stadium, Delhi - DPCC': {   'lat': 28.6125465,

                                                            'lng': 77.2373351},

    'Manali Village, Chennai - TNPCB': {'lat': 13.1779289, 'lng': 80.2700737},

    'Manali, Chennai - CPCB': {'lat': 13.1779289, 'lng': 80.2700737},

    'Mandir Marg, Delhi - DPCC': {'lat': 28.6341752, 'lng': 77.20047459999999},

    'Maninagar, Ahmedabad - GPCB': {'lat': 22.995165, 'lng': 72.604097},

    'Marhatal, Jabalpur - MPPCB': {'lat': 23.1670639, 'lng': 79.9339608},

    'Model Town, Patiala - PPCB': {'lat': 30.3448377, 'lng': 76.3708347},

    'More Chowk Waluj, Aurangabad - MPCB': {   'lat': 19.8406027,

                                               'lng': 75.2466299},

    'Moti Doongri, Alwar - RSPCB': {'lat': 27.5515817, 'lng': 76.6080554},

    'Mundka, Delhi - DPCC': {'lat': 28.6823144, 'lng': 77.034937},

    'Municipal Corporation Office, Dharuhera - HSPCB': {   'lat': 28.2068002,

                                                           'lng': 76.7996532},

    'Muradpur, Patna - BSPCB': {'lat': 25.6194928, 'lng': 85.14663999999999},

    'Murthal, Sonipat - HSPCB': {'lat': 29.0315896, 'lng': 77.0723807},

    'Muzaffarpur Collectorate, Muzaffarpur - BSPCB': {   'lat': 26.1235085,

                                                         'lng': 85.3812437},

    'NISE Gwal Pahari, Gurugram - IMD': {'lat': 28.4235473, 'lng': 77.1489412},

    'NSIT Dwarka, Delhi - CPCB': {'lat': 28.610273, 'lng': 77.0378818},

    'Najafgarh, Delhi - DPCC': {'lat': 28.6090126, 'lng': 76.9854526},

    'Narela, Delhi - DPCC': {'lat': 28.8548818, 'lng': 77.08921509999999},

    'Nathu Colony, Ballabgarh - HSPCB': {   'lat': 28.3426369,

                                            'lng': 77.31772459999999},

    'Nehru Nagar, Delhi - DPCC': {'lat': 28.5638667, 'lng': 77.2608101},

    'Nehru Nagar, Kanpur - UPPCB': {'lat': 26.4715909, 'lng': 80.3237548},

    'Nerul, Navi Mumbai - MPCB': {'lat': 19.0338457, 'lng': 73.0195871},

    'New Collectorate, Baghpat - UPPCB': {   'lat': 28.9427827,

                                             'lng': 77.22760699999999},

    'New Industrial Town, Faridabad - HSPCB': {   'lat': 28.3922002,

                                                  'lng': 77.301675},

    'New Mandi, Muzaffarnagar - UPPCB': {'lat': 29.4676905, 'lng': 77.7115687},

    'Nishant Ganj, Lucknow - UPPCB': {   'lat': 26.8669313,

                                         'lng': 80.94980149999999},

    'North Campus, DU, Delhi - IMD': {'lat': 28.6876514, 'lng': 77.2102816},

    'Okhla Phase-2, Delhi - DPCC': {'lat': 28.5625518, 'lng': 77.2913729},

    'Opp GPO Civil Lines, Nagpur - MPCB': {   'lat': 21.1523552,

                                              'lng': 79.0692636},

    'PWD Grounds, Vijayawada - APPCB': {'lat': 16.5061942, 'lng': 80.6313553},

    'Padmapukur, Howrah - WBPCB': {'lat': 22.5707053, 'lng': 88.3008448},

    'Palayam, Kozhikode - Kerala PCB': {   'lat': 11.2488252,

                                           'lng': 75.78389949999999},

    'Pallavpuram Phase 2, Meerut - UPPCB': {   'lat': 29.0641002,

                                               'lng': 77.7151616},

    'Patparganj, Delhi - DPCC': {'lat': 28.6347308, 'lng': 77.30457109999999},

    'Patti Mehar, Ambala - HSPCB': {'lat': 30.3778718, 'lng': 76.7733263},

    'Peenya, Bengaluru - CPCB': {'lat': 13.0285133, 'lng': 77.5196763},

    'Phase-1 GIDC, Vapi - GPCB': {'lat': 34.1832613, 'lng': -84.2182962},

    'Phase-4 GIDC, Vatva - GPCB': {'lat': 34.1832613, 'lng': -84.2182962},

    'Phool Bagh, Gwalior - Mondelez Ind. Food': {   'lat': 26.2103607,

                                                    'lng': 78.16926889999999},

    'Pimpleshwar Mandir, Thane - MPCB': {   'lat': 19.1890822,

                                            'lng': 72.96224939999999},

    'Plammoodu, Thiruvananthapuram - Kerala PCB': {   'lat': 8.5140567,

                                                      'lng': 76.9477422},

    'Polayathode, Kollam - Kerala PCB': {   'lat': 8.878704899999999,

                                            'lng': 76.6073332},

    'Police Commissionerate, Jaipur - RSPCB': {   'lat': 26.9164092,

                                                  'lng': 75.80167879999999},

    'Police Lines, Jind - HSPCB': {'lat': 29.3069655, 'lng': 76.3478097},

    'Powai, Mumbai - MPCB': {'lat': 19.1175993, 'lng': 72.9059747},

    'Punjab Agricultural University, Ludhiana - PPCB': {   'lat': 30.9010281,

                                                           'lng': 75.8071228},

    'Punjabi Bagh, Delhi - DPCC': {'lat': 28.66197529999999, 'lng': 77.1241557},

    'Pusa, Delhi - DPCC': {'lat': 28.6376724, 'lng': 77.1571443},

    'Pusa, Delhi - IMD': {'lat': 28.6376724, 'lng': 77.1571443},

    'R K Puram, Delhi - DPCC': {'lat': 28.5660075, 'lng': 77.1767435},

    'RIICO Ind. Area III, Bhiwadi - RSPCB': {   'lat': 28.2123547,

                                                'lng': 76.85410739999999},

    'RIMT University, Mandi Gobindgarh - PPCB': {   'lat': 30.6510104,

                                                    'lng': 76.32925200000001},

    'Rabindra Bharati University, Kolkata - WBPCB': {   'lat': 22.5844542,

                                                        'lng': 88.3593841},

    'Rabindra Sarobar, Kolkata - WBPCB': {'lat': 22.5121451, 'lng': 88.3636952},

    'Railway Colony, Guwahati - APCB': {   'lat': 26.1795873,

                                           'lng': 91.78431499999999},

    'Rajbansi Nagar, Patna - BSPCB': {'lat': 25.603603, 'lng': 85.1119721},

    'Ratanpura, Rupnagar - Ambuja Cements': {   'lat': 31.0293638,

                                                'lng': 76.5733862},

    'Rishi Nagar, Kaithal - HSPCB': {   'lat': 29.8029465,

                                        'lng': 76.41436639999999},

    'Rohini, Delhi - DPCC': {'lat': 28.73826769999999, 'lng': 77.0822151},

    'SFTI Kusdihra, Gaya - BSPCB': {'lat': 24.7625227, 'lng': 84.9804538},

    'SIDCO Kurichi, Coimbatore - TNPCB': {'lat': 10.9438095, 'lng': 76.9772675},

    'Sahilara, Maihar - KJS Cements': {   'lat': 24.2609738,

                                          'lng': 80.71866969999999},

    'Samanpura, Patna - BSPCB': {'lat': 25.6074609, 'lng': 85.08456749999999},

    'Sanathnagar, Hyderabad - TSPCB': {'lat': 17.4562544, 'lng': 78.4439295},

    'Sanegurava Halli, Bengaluru - KSPCB': {   'lat': 12.9715987,

                                               'lng': 77.5945627},

    'Sanjay Nagar, Ghaziabad - UPPCB': {'lat': 28.6939957, 'lng': 77.4549679},

    'Sanjay Palace, Agra - UPPCB': {'lat': 27.1986569, 'lng': 78.0059814},

    'Secretariat, Amaravati - APPCB': {'lat': 16.5045615, 'lng': 80.5235168},

    'Sector - 125, Noida - UPPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector - 62, Noida - IMD': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector 11, Faridabad - HSPCB': {'lat': 28.3704165, 'lng': 77.3220128},

    'Sector 30, Faridabad - HSPCB': {'lat': 28.4425186, 'lng': 77.3223915},

    'Sector- 16A, Faridabad - HSPCB': {   'lat': 36.18213,

                                          'lng': -95.78742079999999},

    'Sector-1, Noida - UPPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-10, Gandhinagar - GPCB': {   'lat': 36.18213,

                                         'lng': -95.78742079999999},

    'Sector-116, Noida - UPPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-12, Karnal - HSPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-18, Panipat - HSPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-2 IMT, Manesar - HSPCB': {   'lat': 36.18213,

                                         'lng': -95.78742079999999},

    'Sector-2 Industrial Area, Pithampur - MPPCB': {   'lat': 36.18213,

                                                       'lng': -95.78742079999999},

    'Sector-25, Chandigarh - CPCC': {   'lat': 36.18213,

                                        'lng': -95.78742079999999},

    'Sector-51, Gurugram - HSPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-6, Panchkula - HSPCB': {'lat': 36.18213, 'lng': -95.78742079999999},

    'Sector-7, Kurukshetra - HSPCB': {   'lat': 36.18213,

                                         'lng': -95.78742079999999},

    'Sector-D Industrial Area, Mandideep - MPPCB': {   'lat': 36.18213,

                                                       'lng': -95.78742079999999},

    'Shadipur, Delhi - CPCB': {'lat': 28.651027, 'lng': 77.1562196},

    'Shasthri Nagar, Ratlam - IPCA Lab': {'lat': 23.3312468, 'lng': 75.0432032},

    'Shastri Nagar, Jaipur - RSPCB': {'lat': 26.9503102, 'lng': 75.8009833},

    'Shastri Nagar, Narnaul - HSPCB': {'lat': 28.0636597, 'lng': 76.1116997},

    'Shrinath Puram, Kota - RSPCB': {'lat': 25.136387, 'lng': 75.8246657},

    'Shrivastav Colony, Damoh - MPPCB': {'lat': 23.8184923, 'lng': 79.4338188},

    'Shyam Nagar, Palwal - HSPCB': {   'lat': 28.1482612,

                                       'lng': 77.33316040000001},

    'Sidhu Kanhu Indoor Stadium, Durgapur - WBPCB': {   'lat': 23.5404352,

                                                        'lng': 87.2914112},

    'Sikulpuikawn, Aizawl - Mizoram PCB': {   'lat': 23.7173952,

                                              'lng': 92.7181174},

    'Silk Board, Bengaluru - KSPCB': {'lat': 12.9164812, 'lng': 77.6219055},

    'Sion, Mumbai - MPCB': {'lat': 19.0390214, 'lng': 72.86189519999999},

    'Sirifort, Delhi - CPCB': {'lat': 28.5505827, 'lng': 77.214799},

    'Solapur, Solapur - MPCB': {'lat': 17.6599188, 'lng': 75.9063906},

    'Sonia Vihar, Delhi - DPCC': {'lat': 28.7332472, 'lng': 77.2495891},

    'Sri Aurobindo Marg, Delhi - DPCC': {'lat': 28.5563099, 'lng': 77.2063378},

    'T T Nagar, Bhopal - MPPCB': {'lat': 23.2357524, 'lng': 77.39864709999999},

    'Talcher Coalfields,Talcher - OSPCB': {'lat': 20.9501027, 'lng': 85.216816},

    'Talkatora District Industries Center, Lucknow - CPCB': {   'lat': 26.8332171,

                                                                'lng': 80.8965834},

    'Tata Stadium, Jorapokhar - JSPCB': {'lat': 23.7082799, 'lng': 86.4127228},

    'Teri Gram, Gurugram - HSPCB': {   'lat': 28.4275348,

                                       'lng': 77.14645829999999},

    'Thavakkara, Kannur - Kerala PCB': {   'lat': 11.8701516,

                                           'lng': 75.36905949999999},

    'Tirumala, Tirupati - APPCB': {'lat': 13.6807357, 'lng': 79.3508975},

    'Udyogamandal, Eloor - Kerala PCB': {'lat': 10.0737878, 'lng': 76.3014896},

    'Urban Estate-II, Hisar - HSPCB': {   'lat': 41.885003,

                                          'lng': -87.61686399999999},

    'Urban, Chamarajanagar - KSPCB': {'lat': 11.9271328, 'lng': 76.9326167},

    'Vasai West, Mumbai - MPCB': {'lat': 19.3664631, 'lng': 72.8155136},

    'Vasundhara, Ghaziabad - UPPCB': {'lat': 28.6623758, 'lng': 77.37344},

    'Velachery Res. Area, Chennai - CPCB': {   'lat': 12.9517854,

                                               'lng': 80.2112303},

    'Victoria, Kolkata - WBPCB': {'lat': 22.5448082, 'lng': 88.3425578},

    'Vidayagiri, Bagalkot - KSPCB': {'lat': 16.1756049, 'lng': 75.6586295},

    'Vijay Nagar, Ramanagara - KSPCB': {   'lat': 12.7324268,

                                           'lng': 77.29022660000001},

    'Vikas Sadan, Gurugram - HSPCB': {   'lat': 28.4501238,

                                         'lng': 77.02849379999999},

    'Vile Parle West, Mumbai - MPCB': {'lat': 19.1071283, 'lng': 72.8367535},

    'Vindhyachal STPS, Singrauli - MPPCB': {   'lat': 24.0886334,

                                               'lng': 82.6477523},

    'Vivek Vihar, Delhi - DPCC': {'lat': 28.6712458, 'lng': 77.3176541},

    'Vyttila, Kochi - Kerala PCB': {'lat': 9.968199, 'lng': 76.3182346},

    'Ward-32 Bapupara, Siliguri - WBPCB': {   'lat': 37.123889,

                                              'lng': -95.80261019999999},

    'Wazirpur, Delhi - DPCC': {'lat': 28.69754439999999, 'lng': 77.1604397},

    'Worli, Mumbai - MPCB': {'lat': 18.9986406, 'lng': 72.8173599},

    'Yamunapuram, Bulandshahr - UPPCB': {'lat': 28.4088401, 'lng': 77.8295809},

    'Zoo Park, Hyderabad - TSPCB': {'lat': 17.3537182, 'lng': 78.4399255}}
df_st['Latitude'] = df_st.apply(lambda x: get_coords(x['StationName'])[0],axis=1)

df_st['Longitude'] = df_st.apply(lambda x: get_coords(x['StationName'])[1],axis=1)

df_st.Status.fillna('NA',inplace=True)

px.set_mapbox_access_token(secret_value_1)

fig = px.scatter_mapbox(df_st,

                        lat="Latitude",

                        lon="Longitude",

                        color='Status',

                        color_discrete_sequence=['rgb(119, 221, 119)','rgb(254, 107, 100)','rgb(119, 158, 203)'],

                        mapbox_style='carto-positron',

                        hover_name='StationId',

                        center={"lat": 20.5937, "lon": 78.9629},

                        zoom=3.5,

                        hover_data=['StationName','City','State'],

                        title= 'AQI Stations in India',

#                         width = 300,

#                         height = 500

                       )

fig.update_geos(fitbounds="locations", visible=True)

fig.update_geos(projection_type="orthographic")

fig.update_layout(height=500,width=500,margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor=BGCOLOR)

fig.update_layout(

    legend=dict(

        x=0,

        y=1,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=12,

            color="black"

        ),

        bgcolor="LightSteelBlue",

        bordercolor="Black",

        borderwidth=0

    )

)

fig.show()
group = df_st.groupby(['State','Status'],as_index=False)['StationId'].count()

group.sort_values(['StationId'],inplace=True,ascending=False)

fig = go.Figure()

active = group[group['Status']=='Active']

fig.add_trace(go.Bar(x=active.State,y=active.StationId,name='Active',

                     marker_color='rgb(119, 221, 119)',

                     marker_line_color='black',

                     marker_line_width=1.5, 

                     opacity=0.9

                    ))

na = group[group['Status']=='NA']

fig.add_trace(go.Bar(x=na.State,y=na.StationId,name='NA',

                     marker_color='rgb(119, 158, 203)',

                     marker_line_color='black',

                     marker_line_width=1.5, 

                     opacity=0.9

                    ))

inactive = group[group['Status']=='Inactive']

fig.add_trace(go.Bar(x=inactive.State,y=inactive.StationId,name='Inactive',

                     marker_color='rgb(254, 107, 100)',

                     marker_line_color='black',

                     marker_line_width=1.5, 

                     opacity=0.9

                    ))

fig.update_xaxes(showgrid=False)

fig.update_layout(height=300,template='ggplot2',barmode='stack',title='AQI Stations per City',

                  hovermode='x',

                  paper_bgcolor=BGCOLOR,plot_bgcolor='lightgray',margin=dict(l=20,r=20,t=40,b=20))

fig.show()
df_ind = df_sd.copy()

df_ind['Date'] = pd.to_datetime(df_ind['Date'],format='%Y-%m-%d')

df_ind['Period'] = df_ind.apply(lambda x: 'Before' if (x['Date'] < dt.datetime(2020, 3, 23)) else 'After',axis=1)

df_ind = df_ind.query('Date>="2020-01-01"')

df_ind = df_ind.groupby(['Period','StationId'],as_index=False)['AQI','PM2.5','PM10','O3','CO','SO2','NO2'].mean()

df_ind = df_ind.merge(df_st[['StationId','StationName','State','Latitude','Longitude']],how='inner',on='StationId')
def scale(aqiSeries):

    cmax = aqiSeries.max()

    cmin = aqiSeries.min()

    dt = 1e-5

    good = min((50-cmin)/(cmax-cmin)+dt,1.0)

    satisfactory = min((100-cmin)/(cmax-cmin)+dt,1.0)

    moderate = min((200-cmin)/(cmax-cmin)+dt,1.0)

    poor = min((300-cmin)/(cmax-cmin)+dt,1.0)

    very_poor = min((400-cmin)/(cmax-cmin)+dt,1.0)

    severe = min((500-cmin)/(cmax-cmin)+dt,1.0)



    colorcode = [good,satisfactory,moderate,poor,very_poor,severe]

    colorcode = [0.0 if c<0 else c for c in colorcode]

    colors = ['#77DD77','#33AF13','#F6D20E','#F17700','#FE6B64','#F12424']

    scl = []

    prev = 0

    for i in range(len(colorcode)):

        scl.extend([[prev,colors[i]],[colorcode[i],colors[i]]])

        prev=colorcode[i]

        if colorcode[i]==1.0: break

    if scl[-1][0]!=1.0:

        scl[-1][0]=1.0

    

    return scl
dict_center_zoom={

    'India':[(20.5937,78.9629),2.5],

    'Bengaluru':[(12.9716,77.5946),9],

    'Delhi':[(28.7041,77.1025),8],

    'Mumbai':[(19.0760,72.8777),8],

    'Hyderabad':[(17.3850,78.4867),8],

    'Chennai':[(13.0827,80.2707),9]

}



def draw_aqi_map(df,city):

    if city=='India':

        df0 = df

    else:

        if city=='Bengaluru':

            state='Karnataka'

        elif city=='Mumbai':

            state='Maharashtra'

        elif city=='Hyderabad':

            state='Telangana'

        elif city=='Chennai':

            state='Tamil Nadu'

        else:

            state=city

        df0 = df[df['State']==state]

    



    fig = go.Figure()



    df1=df0[df0['Period']=='Before']

    fig.add_trace(go.Scattermapbox(name='Before Lockdown',

        lat=df1.Latitude,

        lon=df1.Longitude,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=17,

            color=df1.AQI,

            colorscale=scale(df1.AQI),

            opacity=0.7

        ),

        text=df1.StationId.astype(str)+'<br><b>Station</b>: '+df1.StationName+'<br><b>AQI</b>: '+np.round(df1.AQI).astype(str),

        hoverinfo='text',

        subplot='mapbox'

    ))



    df2=df0[df0['Period']=='After']

    fig.add_trace(go.Scattermapbox(name='After Lockdown',

        lat=df2.Latitude,

        lon=df2.Longitude,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=17,

            color=df2.AQI,

            colorscale=scale(df2.AQI),

            opacity=0.7

        ),

        text=df2.StationId.astype(str)+'<br><b>Station</b>: '+df2.StationName+'<br><b>AQI</b>: '+np.round(df2.AQI).astype(str),

        hoverinfo='text',

        subplot='mapbox2'

    ))



    fig.update_layout(

        height=300,width=600,

        title=city + ': Before & After Lockdown',

        paper_bgcolor=BGCOLOR,

        margin=dict(l=20,r=20,t=40,b=20),

        showlegend=False,

        autosize=True,

        hovermode='closest',

        mapbox=dict(accesstoken=secret_value_1,

            style='carto-positron',

            domain={'x': [0, 0.48], 'y': [0, 1]},

                bearing=0,

                center=dict(

                lat=dict_center_zoom[city][0][0],

                lon=dict_center_zoom[city][0][1]

            ),

        pitch=0,

        zoom=dict_center_zoom[city][1]

        ),

        mapbox2=dict(accesstoken=secret_value_1,

            style='carto-positron',

            domain={'x': [0.52, 1.0], 'y': [0, 1]},

            bearing=0,

            center=dict(

                lat=dict_center_zoom[city][0][0],

                lon=dict_center_zoom[city][0][1]

            ),

            pitch=0,

            zoom=dict_center_zoom[city][1],

        ),

    )

    return fig
draw_aqi_map(df_ind,'India')
draw_aqi_map(df_ind,'Delhi')
draw_aqi_map(df_ind,'Bengaluru')
draw_aqi_map(df_ind,'Mumbai')
draw_aqi_map(df_ind,'Hyderabad')
draw_aqi_map(df_ind,'Chennai')
df_temp_all = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

df_temp = df_temp_all[df_temp_all['Country']=='India']

#Rename some values

df_temp.loc[df_temp['City']=='Bombay (Mumbai)','City']='Mumbai'

df_temp.loc[df_temp['City']=='Calcutta','City']='Kolkata'

df_temp.loc[df_temp['City']=='Chennai (Madras)','City']='Chennai'

#Create a date column

df_temp['Date']=df_temp.apply(lambda x: str(dt.date(x.Year,x.Month,x.Day)),axis=1)

#Convert from fahrenheit to celsius (°F - 32) x 5/9 = °C 

df_temp['AvgTemperature'] = df_temp['AvgTemperature'].apply(lambda x: np.round(((x-32)*5)/9))

df_temp = df_temp[['Date','City','AvgTemperature']]

df_temp.rename(columns={'AvgTemperature':'Temp'},inplace=True)

#Merge the data

df_metro = pd.merge(df_cd,df_temp,how='inner',on=['City','Date'])

df_metro = df_metro[['City','Date','PM2.5','PM10','O3','NO2','SO2','CO','Temp']]

df_metro = df_metro.melt(id_vars=['City','Date'], var_name='Metric', value_name='Value')

df_metro.columns.name=''

df_metro.loc[df_metro['Value']==-73,'Value'] = np.float('nan')
fig = go.Figure()

cities=df_metro.City.unique()

metric_color={

    'PM2.5':'rgb(66, 133, 244)',

    'PM10':'rgb(234, 67, 53)',

    'O3':'rgb(173, 100, 100)',

    'NO2':'rgb(110, 27, 9)',

    'SO2':'rgb(57, 58, 60)',

    'CO':'rgb(240, 114, 73)',

    'Temp':'rgb(52, 168, 83)'

}

#create the dropdown

buttons=[]

metrics = ['Temp','CO','SO2','NO2','O3','PM10','PM2.5']

for i,city in enumerate(cities):

    if i == 0:

        visible=True

    else:

        visible=False

    group_city = df_metro[df_metro['City']==city]

    for idx,metric in enumerate(metrics):

        group_city_metric = group_city[group_city['Metric']==metric]

        fig.add_trace(go.Scatter(name=metric,

                                 x=group_city_metric['Date'],

                                 y=group_city_metric['Value'],

                                 fill='tozeroy',

                                 visible=visible,

                                 line_color=metric_color[metric],

                                 yaxis='y'+str(idx+1) if idx+1!=1 else 'y'))

    dic = dict(label='',

               method="update",

               args=[{"visible": [False]*len(cities)},{"title": ''}])

    dic['label']=city

    dic['args'][0]['visible'][i] = True

    dic['args'][1]['title'] = 'Condition of '+city

    buttons.append(dic)

fig.update_layout(paper_bgcolor=BGCOLOR,plot_bgcolor='lightgray',

                      updatemenus=[dict(

                          active=0,

                          bgcolor='rgb(250, 250, 255)',

                          buttons=buttons,

                      )])

fig.update_layout(

        xaxis=dict(

        #autorange=True,

        range = ['2015-01-01','2020-05-01'],

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(count=2,

                     label="2y",

                     step="year",

                     stepmode="backward"),

                dict(count=3,

                     label="3y",

                     step="year",

                     stepmode="backward"),

                dict(count=4,

                     label="4y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            autorange=True,

        ),

        type="date"

    ),

    yaxis=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[0:2],

        mirror=True,

        showline=True,

        side="right",

        tickfont={"size":10},

        tickmode="auto",

        ticks="",

        title='Temp.',

        titlefont={"size":20},

        type="linear",

        zeroline=False

    ),

    yaxis2=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[1:3],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks="",

        title = 'CO',

        titlefont={"size":20},

        type="linear",

        zeroline=False

    ),

    yaxis3=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[2:4],

        mirror=True,

        showline=True,

        side="right",

        tickfont={"size":10},

        tickmode="auto",

        ticks='',

        title="SO2",

        titlefont={"size":20},

        type="linear",

        zeroline=False

    ),

    yaxis4=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[3:5],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks='',

        title="NO2",

        titlefont={"size":20},

        type="linear",

        zeroline=True

    ),

    yaxis5=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[4:6],

        mirror=True,

        showline=True,

        side="right",

        tickfont={"size":10},

        tickmode="auto",

        ticks='',

        title="O3",

        titlefont={"size":20},

        type="linear",

        zeroline=True

    ),

    yaxis6=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[5:7],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks='',

        title="PM10",

        titlefont={"size":20},

        type="linear",

        zeroline=True

    ),

    yaxis7=dict(

        anchor="x",

        autorange=True,

        domain=np.linspace(0,1,8).tolist()[6:8],

        mirror=True,

        showline=True,

        side="right",

        tickfont={"size":10},

        tickmode='array',

        ticks='',

        title="PM2.5",

        titlefont={"size":20},

        type="linear",

        zeroline=True

    )

    )

fig.update_layout(margin=dict(l=20,r=20,t=70,b=20),template='seaborn', 

                  title='Condition of Chennai',showlegend=False)

fig.show()
def build_city_plot(city):

    df = df_cd[df_cd['City']==city]

    fig = go.Figure()

    cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2','O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

    buttons = []

    for idx,col in enumerate(cols): 

        fig.add_trace(go.Scatter(name=col,x=df['Date'],y=df[col]))

        dic = dict(label='',

                   method="update",

                   args=[{"visible": [False]*len(cols)},{"title": ''}])

        dic['label']=col

        dic['args'][0]['visible'][idx] = True

        dic['args'][1]['title'] = city +' ('+col+')'

        buttons.append(dic)



    #Add one case to display all the metrics at once

    all_params = dict(label='All',

                      method="update",

                      args=[{"visible": [True]*len(cols)},{"title": city +' (All)'}])

    buttons.append(all_params)



    fig.update_layout(paper_bgcolor=BGCOLOR,

                      updatemenus=[dict(

                          active=13,

                          bgcolor='rgb(31, 119, 180)',

                          buttons=buttons,

                          direction='right',

                          font=dict(color='black')

                      )])

    fig.update_xaxes(rangeslider_visible=True,

                 rangeselector=dict(buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward",),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ]),bgcolor='rgb(31, 119, 180)',font=dict(color='black')))

    fig.update_layout(title= city+' (All)',height=500,margin=dict(b=20))

    fig.show()
def build_calmap(df,city=''):

    if city:

        df = df[df['City']==city]

    else:

        city = 'India'

    df.drop(df[pd.isnull(df['AQI'])].index,axis=0,inplace=True)

    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

    df['Year'] = df.Date.dt.year

    df['Day'] = df.Date.dt.day_name()

    df['Week'] = df.Date.dt.week

    df['avg_AQI'] = df.groupby(['Year','Week','Day'])['AQI'].transform('mean')

    years = df.Year.unique().tolist()

    fig = make_subplots(rows=len(years),cols=1,shared_xaxes=True, vertical_spacing=0.005)

    r = 1

    for year in years:

        df_year = df[df['Year']==year]

        fig.add_trace(go.Heatmap(

            name=year,

            z=df_year['avg_AQI'],

            x=df_year.Week,

            y=df_year.Day,

            coloraxis = 'coloraxis'

        ),r,1)

        fig.update_yaxes(title_text=year,tickfont=dict(size=5),row=r,col=1)

        r+=1

    fig.update_xaxes(range=[0,53],tickfont=dict(size=10), nticks=53)

    fig.update_layout(coloraxis = {'colorscale':scale(df['avg_AQI'])})

    fig.update_layout(paper_bgcolor=BGCOLOR,plot_bgcolor='lightgrey',title='<b>'+city+'</b>' + ': Variation of AQI over the years',

                      margin=dict(t=35, b=20))

    return fig
fig = build_calmap(df_cd)

fig.show()
fig = build_calmap(df_cd,city='Delhi')

fig.show()
fig = build_calmap(df_cd,city='Bengaluru')

fig.show()
fig = build_calmap(df_cd,city='Chennai')

fig.show()
fig = build_calmap(df_cd,city='Mumbai')

fig.show()
fig = build_calmap(df_cd,city='Kolkata')

fig.show()
def build_calmap_month(df,city=''):

    if city:

        df = df[df['City']==city]

    else:

        city = 'India'

    df.drop(df[pd.isnull(df['AQI'])].index,axis=0,inplace=True)

    df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

    df['Year'] = df.Date.dt.year

    df['Month'] = df.Date.dt.month_name()

    df['Day'] = df.Date.dt.day

    df['avg_AQI'] = df.groupby(['Year','Month','Day'])['AQI'].transform('mean')

    years = df.Year.unique().tolist()

    fig = make_subplots(rows=len(years),cols=1,shared_xaxes=True, vertical_spacing=0.005)

    r = 1

    for year in years:

        df_year = df[df['Year']==year]

        fig.add_trace(go.Heatmap(

            z=df_year['avg_AQI'],

            x=df_year.Day,

            y=df_year.Month,

            coloraxis = 'coloraxis'

        ),r,1)

        fig.update_yaxes(title_text=year,tickfont=dict(size=5),row=r,col=1)

        r+=1

    fig.update_xaxes(range=[0,31],tickfont=dict(size=10), nticks=31)

    fig.update_layout(coloraxis = {'colorscale':scale(df['avg_AQI'])})

    fig.update_layout(paper_bgcolor=BGCOLOR,plot_bgcolor='lightgrey',title='<b>'+city+'</b>' + ': Variation of AQI monthly over the years',

                      margin=dict(t=35, b=20))

    return fig
fig = build_calmap_month(df_cd)

fig.show()
fig = build_calmap_month(df_cd,'Delhi')

fig.show()
fig = build_calmap_month(df_cd,'Bengaluru')

fig.show()
fig = build_calmap_month(df_cd,'Chennai')

fig.show()
fig = build_calmap_month(df_cd,'Mumbai')

fig.show()
fig = build_calmap_month(df_cd,'Kolkata')

fig.show()
def build_calmap_hourly(df,city=''):

    if city:

        df = df[df['City']==city]

    else:

        city = 'India'

    df.drop(df[pd.isnull(df['AQI'])].index,axis=0,inplace=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df['Year'] = df.Datetime.dt.year

    df['Hour'] = df.Datetime.dt.hour

    df['Dayofyear'] = df.Datetime.dt.dayofyear

    df['avg_AQI'] = df.groupby(['Year','Dayofyear','Hour'])['AQI'].transform('mean')

    years = df.Year.unique().tolist()

    fig = make_subplots(rows=len(years),cols=1,shared_xaxes=True, vertical_spacing=0.005)

    r = 1

    for year in years:

        df_year = df[df['Year']==year]

        fig.add_trace(go.Heatmap(

            z=df_year['avg_AQI'],

            x=df_year.Dayofyear,

            y=df_year.Hour,

            coloraxis = 'coloraxis'

        ),r,1)

        fig.update_yaxes(range=[0,23], nticks=12,title_text=year,tickfont=dict(size=5),row=r,col=1)

        r+=1

    fig.update_xaxes(range=[0,366],tickfont=dict(size=5), nticks=180)

    fig.update_layout(coloraxis = {'colorscale':scale(df['avg_AQI'])})

    fig.update_layout(paper_bgcolor=BGCOLOR,plot_bgcolor='lightgrey',title='<b>'+city+'</b>' + ': Variation of AQI monthly over the years',

                      margin=dict(t=35, b=20))

    return fig
fig=build_calmap_hourly(df_ch)

fig.show()
fig=build_calmap_hourly(df_ch,'Delhi')

fig.show()
fig=build_calmap_hourly(df_ch,'Bengaluru')

fig.show()
fig=build_calmap_hourly(df_ch,'Chennai')

fig.show()
fig=build_calmap_hourly(df_ch,'Mumbai')

fig.show()
fig=build_calmap_hourly(df_ch,'Kolkata')

fig.show()