import pandas as pd

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

import numpy as np

import warnings 
#Filtering warning messages

warnings.simplefilter("ignore")
#Fetching data

data = pd.read_csv("../input/MissingMigrants-Global-2019-03-29T18-36-07.csv")
#Displaying columns

data.columns
#Renaming columns

data.rename(columns={'Region of Incident':'Region_of_Incident','Reported Date':'Reported_Date','Reported Year':'Reported_Year','Reported Month':'Reported_Month',

                     'Number Dead':'Number_Dead', 'Minimum Estimated Number of Missing':'Minimum_Estimated_Number_of_Missing',

                    'Total Dead and Missing':'Total_Dead_and_Missing','Number of Survivors':'Number_of_Survivors',

                    'Number of Females':'Number_of_Females','Number of Males':'Number_of_Males','Web ID':'Web_ID',

                    'Number of Children':'Number_of_Children', 'Cause of Death':'Cause_of_Death',

                    'Location Description':'Location_Description', 'Information Source':'Information_Source',

                    'Location Coordinates':'Location_Coordinates','Migration Route':'Migration_Route',

                    'UNSD Geographical Grouping':'UNSD_Geographical_Grouping','Source Quality':'Source_Quality'}, inplace=True)
#Displaying the description of data

data.describe()
#Peek at data

data.head(5)
#Displaying total no. of empty cells 

data.isna().sum()
#Dropping specific columns NaNs

data = data.dropna(subset = ['Location_Coordinates'])
#Extracting region specific data

reg1 = data[data['Region_of_Incident']=='Caribbean']

reg2 = data[data['Region_of_Incident']=='Central America']

reg3 = data[data['Region_of_Incident']=='Central Asia']

reg4 = data[data['Region_of_Incident']=='East Asia']

reg5 = data[data['Region_of_Incident']=='Europe']

reg6 = data[data['Region_of_Incident']=='Horn of Africa']

reg7 = data[data['Region_of_Incident']=='Mediterranean']

reg8 = data[data['Region_of_Incident']=='Middle East']

reg9 = data[data['Region_of_Incident']=='North Africa']

reg10 = data[data['Region_of_Incident']=='North America']

reg11 = data[data['Region_of_Incident']=='South America']

reg12 = data[data['Region_of_Incident']=='South Asia']

reg13 = data[data['Region_of_Incident']=='Southeast Asia']

reg14 = data[data['Region_of_Incident']=='Sub-Saharan Africa']

reg15 = data[data['Region_of_Incident']=='US-Mexico Border']
#Region specific countries location details

reg1_loc = [[23.6260,-102.5375,'Mexico'], [14.7504,-86.2413,'Honduras'], [12.8692,-85.1412,'Nicaragua'], 

            [21.5513, -79.6017, 'Cuba'], [18.7009047, -70.1654548, 'Republica Dominicana'],

            [5.1633, -69.4147, 'Venezuela'], [4.1157, -72.9301, 'Colombia'], [8.4255, -80.1054, 'Panama'],

            [39.381266, -97.922211, 'USA']]

reg2_loc =[[17.189877, -88.49765,'Belize'], [9.748917, -83.753428, 'Costa Rica'],

           [13.794185, -88.89653, 'El Salvador'], [15.783471, -90.230759, 'Guatemala'],

           [15.199999, -86.241905, 'Honduras'], [12.8692,-85.1412,'Nicaragua'],

           [8.4255, -80.1054, 'Panama'], [23.6260,-102.5375,'Mexico']] 

reg3_loc =[[48.019573, 66.923684, 'Kazakhstan'], [41.20438, 74.766098, 'Kyrgyzstan'],

           [38.861034, 71.276093, 'Tajikistan'],[38.969719, 59.556278, 'Turkmenistan'],

           [41.377491, 64.585262, 'Uzbekistan']]

reg4_loc =[[35.86166, 104.195397, 'China'], [22.396428, 114.109497, 'Hong Kong'],

           [36.204824, 138.252924, 'Japan'],[46.862496, 103.846656, 'Mongolia'],[40.339852, 127.510093, 'North Korea'],

           [35.907757, 127.766922, 'South Korea'],[23.69781, 120.960515, 'Taiwan']]

reg5_loc =[[61.52401, 105.318756, 'Russia'], [51.165691, 10.451526, 'Germany'], [55.378051, -3.435973, 'United Kingdom'],[46.227638, 2.213749, 'France'],[41.87194, 12.56738, 'Italy'],

           [40.463667, -3.74922, 'Spain'], [48.379433, 31.16558, 'Ukraine'], [51.919438, 19.145136, 'Poland'],[45.943161, 24.96676, 'Romania'],[52.132633, 5.291266, 'Netherlands'],

           [50.503887, 4.469936, 'Belgium'], [39.074208, 21.824312, 'Greece'], [49.817492, 15.472962, 'Czech Republic'],[39.399872, -8.224454, 'Portugal'],[60.128161, 18.643501, 'Sweden'],

           [47.162494, 19.503304, 'Hungary'], [53.709807, 27.953389, 'Belarus'], [47.516231, 14.550072, 'Austria'],[44.016521, 21.005859, 'Serbia'],[46.818188, 8.227512, 'Switzerland'],

           [42.733883, 25.48583, 'Bulgaria'], [56.26392, 9.501785, 'Denmark'], [61.92411, 25.748151, 'Finland'],[48.669026, 19.699024, 'Slovakia'],[60.472024, 8.468946, 'Norway'],

           [53.41291, -8.24389, 'Ireland'], [45.1, 15.2, 'Croatia'], [47.411631, 28.369885, 'Moldova'],[43.915886, 17.679076, 'Bosnia and Herzegovina'],[41.153332, 20.168331, 'Albania'],

           [55.169438, 23.881275, 'Lithuania'], [41.608635, 21.745275, 'Macedonia [FYROM]'], [46.151241, 14.995463, 'Slovenia'],[56.879635, 24.603189, 'Latvia'],[58.595272, 25.013607, 'Estonia'],

           [42.708678, 19.37439, 'Montenegro'], [49.815273, 6.129583, 'Luxembourg'], [35.937496, 14.375416, 'Malta'],[64.963051, -19.020835, 'Iceland'],[42.546245, 1.601554, 'Andorra'],

           [43.750298, 7.412841, 'Monaco'], [47.166, 9.555373, 'Liechtenstein'], [43.94236, 12.457777, 'San Marino']]

reg6_loc = [[11.825138, 42.590275, 'Djibouti'], [15.179384, 39.782334, 'Eritrea'],

            [9.145, 40.489673, 'Ethiopia'], [5.152149, 46.199616, 'Somalia'],

            [-0.023559, 37.906193, 'Kenya']]

reg7_loc =[[26.820553, 30.802498, 'Egypt'], [38.963745, 35.243322, 'Turkey'], [46.227638, 2.213749, 'France'],[41.87194, 12.56738, 'Italy'],[40.463667, -3.74922, 'Spain'],

           [28.033886, 1.659626, 'Algeria'], [31.791702, -7.09262, 'Morocco'], [34.802075, 38.996815, 'Syria'],[33.886917, 9.537499, 'Tunisia'],[39.074208, 21.824312, 'Greece'],

           [31.046051, 34.851612, 'Israel'], [33.854721, 35.862285, 'Lebanon'], [26.3351, 17.228331, 'Libya'],[31.952162, 35.233154, 'Palestinian Territories'],[45.1, 15.2, 'Croatia'],

           [43.915886, 17.679076, 'Bosnia and Herzegovina'], [41.153332, 20.168331, 'Albania'], [46.151241, 14.995463, 'Slovenia'],[35.126413, 33.429859, 'Cyprus'],[42.708678, 19.37439, 'Montenegro'],

           [35.937496, 14.375416, 'Malta'], [43.750298, 7.412841, 'Monaco'], [36.137741, -5.345374, 'Gibraltar']]

reg8_loc =[[25.930414, 50.637772, 'Bahrain'], [35.126413, 33.429859, 'Cyprus'], [26.820553, 30.802498, 'Egypt'],[32.427908, 53.688046, 'Iran'],[33.223191, 43.679291, 'Iraq'],

           [31.046051, 34.851612, 'Israel'], [30.585164, 36.238414, 'Jordan'], [29.31166, 47.481766, 'Kuwait'],[33.854721, 35.862285, 'Lebanon'],[21.512583, 55.923255, 'Oman'],

           [25.354826, 51.183884, 'Qatar'], [23.885942, 45.079162, 'Saudi Arabia'], [34.802075, 38.996815, 'Syria'],[38.963745, 35.243322, 'Turkey'],[23.424076, 53.847818, 'United Arab Emirates'],

           [15.552727, 48.516388, 'Yemen']]

reg9_loc = [[28.033886, 1.659626, 'Algeria'], [26.820553, 30.802498, 'Egypt'], [26.3351, 17.228331, 'Libya'], [31.791702, -7.09262, 'Morocco'], [12.862807, 30.217636, 'Sudan'], 

            [33.886917, 9.537499, 'Tunisia'], [24.215527, -12.885834, 'Western Sahara'],[15.454166, 18.732207, 'Chad'], [17.607789, 8.081666, 'Niger'],

            [17.570692, -3.996166, 'Mali'], [9.081999, 8.675277, 'Nigeria'],[9.145, 40.489673, 'Ethiopia']]

reg10_loc =[[17.060816, -61.796428, 'Antigua and Barbuda'], [25.03428, -77.39628, 'Bahamas'], [13.193887, -59.543198, 'Barbados'],[17.189877, -88.49765, 'Belize'],

            [9.748917, -83.753428, 'Costa Rica'], [21.521757, -77.781167, 'Cuba'], [18.735693, -70.162651, 'Dominican Republic'],[13.794185, -88.89653, 'El Salvador'],[12.262776, -61.604171, 'Grenada'],

            [15.783471, -90.230759, 'Guatemala'], [18.971187, -72.285215, 'Haiti'], [15.199999, -86.241905, 'Honduras'],[18.109581, -77.297508, 'Jamaica'],[23.634501, -102.552784, 'Mexico'],

            [12.865416, -85.207229, 'Nicaragua'], [8.537981, -80.782127, 'Panama'], [17.357822, -62.782998, 'Saint Kitts and Nevis'],[13.909444, -60.978893, 'Saint Lucia'],

            [12.984305, -61.287228, 'Saint Vincent and the Grenadines'],[10.691803, -61.222503, 'Trinidad and Tobago'], [37.09024, -95.712891, 'United States']]

reg11_loc = [[-38.416097, -63.616672, 'Argentina'], [-16.290154, -63.588653, 'Bolivia'], [-14.235004, -51.92528, 'Brazil'], [-35.675147, -71.542969, 'Chile'], [4.570868, -74.297333, 'Colombia'], 

             [-1.831239, -78.183406, 'Ecuador'], [4.860416, -58.93018, 'Guyana'],[-23.442503, -58.443832, 'Paraguay'], [-9.189967, -75.015152, 'Peru'],

             [3.919305, -56.027783, 'Suriname'], [-32.522779, -55.765835, 'Uruguay'],[6.42375, -66.58973, 'Venezuela']]

reg12_loc = [[33.93911, 67.709953, 'Afghanistan'], [23.684994, 90.356331, 'Bangladesh'], [27.514162, 90.433601, 'Bhutan'], 

             [20.593684, 78.96288, 'India'], [3.202778, 73.22068, 'Maldives'],[28.394857, 84.124008, 'Nepal'],

             [30.375321, 69.345116, 'Pakistan'], [7.873054, 80.771797, 'Sri Lanka'], [32.427908, 53.688046, 'Iran'],

             [7.873054, 80.771797, 'Sri Lanka'], [32.427908, 53.688046, 'Iran']]

reg13_loc = [[4.535277, 114.727669, 'Brunei'], [21.913965, 95.956223, 'Myanmar [Burma]'], [12.565679, 104.990963, 'Cambodia'], 

             [-8.874217, 125.727539, 'Timor-Leste'], [-0.789275, 113.921327, 'Indonesia'],[19.85627, 102.495496, 'Laos'],

             [4.210484, 101.975766, 'Malaysia'], [12.879721, 121.774017, 'Philippines'], [1.352083, 103.819836, 'Singapore'],

             [15.870032, 100.992541, 'Thailand'], [14.058324, 108.277199, 'Vietnam']]

reg14_loc =[[-11.202692, 17.873887, 'Angola'], [9.30769, 2.315834, 'Benin'], [-22.328474, 24.684866, 'Botswana'],[12.238333, -1.561593, 'Burkina Faso'],[-3.373056, 29.918886, 'Burundi'],

            [7.369722, 12.354722, 'Cameroon'], [6.611111, 20.939444, 'Central African Republic'],[15.454166, 18.732207, 'Chad'],[-11.875001, 43.872219, 'Comoros'],

            [-4.038333, 21.758664, 'Congo [DRC]'], [-0.228021, 15.827659, 'Congo [Republic]'], [7.539989, -5.54708, 'Côte dIvoire'],[11.825138, 42.590275, 'Djibouti'],[1.650801, 10.267895, 'Equatorial Guinea'],

            [15.179384, 39.782334, 'Eritrea'], [9.145, 40.489673, 'Ethiopia'], [-0.803689, 11.609444, 'Gabon'],[13.443182, -15.310139, 'Gambia'],[7.946527, -1.023194, 'Ghana'],

            [9.945587, -9.696645, 'Guinea'], [11.803749, -15.180413, 'Guinea-Bissau'], [-0.023559, 37.906193, 'Kenya'],[-29.609988, 28.233608, 'Lesotho'],[6.428055, -9.429499, 'Liberia'],

            [-18.766947, 46.869107, 'Madagascar'], [-13.254308, 34.301525, 'Malawi'], [17.570692, -3.996166, 'Mali'],[21.00789, -10.940835, 'Mauritania'],

            [-18.665695, 35.529562, 'Mozambique'], [-22.95764, 18.49041, 'Namibia'], [17.607789, 8.081666, 'Niger'],[9.081999, 8.675277, 'Nigeria'],

            [-1.940278, 29.873888, 'Rwanda'], [14.497401, -14.452362, 'Senegal'],[8.460555, -11.779889, 'Sierra Leone'],[5.152149, 46.199616, 'Somalia'],

            [-30.559482, 22.937506, 'South Africa'], [12.862807, 30.217636, 'Sudan'], [-26.522503, 31.465866, 'Swaziland'], [-6.369028, 34.888822, 'Tanzania'], [8.619543, 0.824782, 'Togo'],

            [1.373333, 32.290275, 'Uganda'], [24.215527, -12.885834, 'Western Sahara'],[-13.133897, 27.849332, 'Zambia'], [-19.015438, 29.154857, 'Zimbabwe']]
#Region - Caribbean

fig = plt.figure(figsize=(10, 10))

plt.title('Migration Incidents in Caribbean Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=14.639197, lon_0=-75.139594,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg1['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg1_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - Central America

fig = plt.figure(figsize=(15, 15))

plt.title('Migration Incidents in Central America Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=14.526309, lon_0=-88.291147,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg2['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg2_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=10);
#Region - Central Asia

fig = plt.figure(figsize=(10, 10))

plt.title('Migration Incidents in Central Asia Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=45.450688, lon_0=68.831901,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg3['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=5)

for xloc, yloc, cnme in reg3_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=10);
#Region - East Asia

fig = plt.figure(figsize=(10, 10))

plt.title('Migration Incidents in East Asia Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=38.794595, lon_0=106.534838,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg4['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg4_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=9);
#Region - Europe

fig = plt.figure(figsize=(30, 30))

plt.title('Migration Incidents in Europe Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=54.525961, lon_0=15.255119,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg5['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=4)

for xloc, yloc, cnme in reg5_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=15);
#Region - Horn of Africa

fig = plt.figure(figsize=(12, 12))

plt.title('Migration Incidents in Horn of Africa Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=9.130378, lon_0=41.280858,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg6['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg6_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - Mediterranean 

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in Mediterranean Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=34.5531284, lon_0=18.048010500000032,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg7['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=1)

for xloc, yloc, cnme in reg7_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - Middle East 

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in Middle East Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=29.298528, lon_0=42.55096,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg8['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=1)

for xloc, yloc, cnme in reg8_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - North Africa 

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in North Africa Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=23.416203, lon_0=25.66283,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg9['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=1)

for xloc, yloc, cnme in reg9_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - North America

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in North America Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=14.639197, lon_0=-75.139594,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg10['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=5)

for xloc, yloc, cnme in reg10_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - South America

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in South America Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=-8.783195, lon_0=-55.491477,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg11['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg11_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - South Asia

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in South Asia Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=25.03764, lon_0=76.456309,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg12['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg12_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - SouthEast Asia

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in SouthEast Asia Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=-2.21797, lon_0=115.66283,)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg13['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=2)

for xloc, yloc, cnme in reg13_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - Sub-Saharan Africa

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in Sub-Saharan Africa Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=3.211057, lon_0=20.880843)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg14['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=1)

for xloc, yloc, cnme in reg14_loc:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Region - US-Mexico Border

fig = plt.figure(figsize=(14, 14))

plt.title('Migration Incidents in US-Mexico Border Region')

m = Basemap(projection='lcc', resolution=None,

            width=8E6, height=8E6, 

            lat_0=29.882858, lon_0=-104.451189)

m.etopo(scale=0.5, alpha=0.5)

for sNo, loc in reg15['Location_Coordinates'].iteritems():

    xloc, yloc = loc.split(',')

    x, y = m(yloc, xloc)

    m.plot(x, y, 'ok', markersize=1)

for xloc, yloc, cnme in [[23.6260,-102.5375,'Mexico'],[39.381266, -97.922211, 'USA']]:

    x, y = m(yloc, xloc)

    m.plot(x, y, 'co', markersize=4)

    plt.text(x, y, cnme, fontsize=11);
#Total No. of Deaths/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Number_Dead.sum().plot('barh')

plt.xlabel('Total Deaths', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Total No. of Deaths/Region', fontsize=15)

plt.show()
#Minimum Estimated Number of Missing/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Minimum_Estimated_Number_of_Missing.sum().plot('barh')

plt.xlabel('Minimum Estimated Number of Missing Persons', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Minimum Estimated Number of Missing Persons/Region', fontsize=15)

plt.show()
#Total Dead and Missing Persons/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Total_Dead_and_Missing.sum().plot('barh')

plt.xlabel('Total Dead and Missing Persons', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Total Dead and Missing Persons/Region', fontsize=15)

plt.show()
#Number of Survivors/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Number_of_Survivors.sum().plot('barh')

plt.xlabel('Number of Survivors', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Number of Survivors/Region', fontsize=15)

plt.show()
#Number of Females Involved in the Migration Incidents/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Number_of_Females.sum().plot('barh')

plt.xlabel('Number of Females Involved in the Migration Incidents', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Number of Females Involved in the Migration Incidents/Region', fontsize=15)

plt.show()
#Number of Males Involved in the Migration Incidents/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Number_of_Males.sum().plot('barh')

plt.xlabel('Number of Males Involved in the Migration Incidents', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Number of Males Involved in the Migration Incidents/Region', fontsize=15)

plt.show()
#Number of Children Involved in the Migration Incidents/Region

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,10))

data.groupby(['Region_of_Incident']).Number_of_Children.sum().plot('barh')

plt.xlabel('Number of Children Involved in the Migration Incidents', fontsize=13)

plt.ylabel('Incident Region', fontsize=13)

plt.title('Number of Children Involved in the Migration Incidents/Region', fontsize=15)

plt.show()
#Cause of Deaths in the Migration Incidents

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(10,100))

data.groupby(['Cause_of_Death']).Web_ID.count().plot('barh')

plt.xlabel('Total No. of Deaths', fontsize=13)

plt.ylabel('Cause of Deaths', fontsize=13)

plt.title('Cause of Deaths in the Migration Incidents', fontsize=15)

plt.show()
#Total Deaths occured in the incident regions in 2014

ry14 = np.array(data[data['Reported_Year']==2014].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry14)

pl.figure(figsize =(20,15))

N = len(ry14)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry14, width, label='Deaths - 2014')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2014', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','East Asia','Europe','Horn of Africa','Mediterranean','Middle East','North Africa','South Asia','Southeast Asia','Sub-Saharan Africa','US-Mexico Border'))

plt.legend(loc='best')

plt.show()
#Total Deaths occured in the incident regions in 2015

ry15 = np.array(data[data['Reported_Year']==2015].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry15)

pl.figure(figsize =(20,15))

N = len(ry15)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry15, width, label='Deaths - 2015')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2015', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','East Asia','Europe','Horn of Africa','Mediterranean','Middle East','North Africa', 'South America','South Asia','Southeast Asia','Sub-Sahrn Africa','US-Mex Border'))

plt.legend(loc='best')

plt.show()
#Total Deaths occured in the incident regions in 2016

ry16 = np.array(data[data['Reported_Year']==2016].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry16)

pl.figure(figsize =(20,15))

N = len(ry16)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry16, width, label='Deaths - 2016')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2016', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','East Asia','Europe','Horn of Africa','Mediterranean','Middle East','North Africa', 'South America','South Asia','Southeast Asia','Sub-Sahrn Africa','US-Mex Border'))

plt.legend(loc='best')

plt.show()
#Total Deaths occured in the incident regions in 2017

ry17 = np.array(data[data['Reported_Year']==2017].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry17)

pl.figure(figsize =(20,15))

N = len(ry17)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry17, width, label='Deaths - 2017')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2017', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','East Asia','Europe','Horn of Africa','Mediterranean','Middle East','North Africa', 'South America','South Asia','Southeast Asia','Sub-Sahrn Africa','US-Mex Border'))

plt.legend(loc='best')

plt.show()
#Total Deaths occured in the incident regions in 2018

ry18 = np.array(data[data['Reported_Year']==2018].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry18)

pl.figure(figsize =(20,15))

N = len(ry18)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry18, width, label='Deaths - 2018')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2018', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','Central Asia','East Asia','Europe','Horn of Africa','Mediterranean','Middle East','North Africa', 'South America','South Asia','Southeast Asia','Sub-Sahrn Africa','US-Mex Border'))

plt.legend(loc='best')

plt.show()
#Total Deaths occured in the incident regions in 2019

ry19 = np.array(data[data['Reported_Year']==2019].groupby(['Region_of_Incident']).Number_Dead.sum())

#print(ry19)

pl.figure(figsize =(20,15))

N = len(ry19)

ind = np.arange(N)

width = 0.10       

plt.bar(ind, ry19, width, label='Deaths - 2019')

plt.ylabel('Total Deaths', fontsize=15)

plt.xlabel('Region of Incident', fontsize=15)

plt.title('Total Deaths occured in the incident regions in 2019', fontsize=15)



plt.xticks(ind + width/6, ('Caribbean','Central America','Europe','Horn of Africa','Mediterranean','Middle East','North Africa', 'South America','Southeast Asia','US-Mex Border'))

plt.legend(loc='best')

plt.show()