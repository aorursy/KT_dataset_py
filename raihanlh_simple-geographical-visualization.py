import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re

import folium

from folium import plugins
data = pd.read_csv("../input/list-faskes-bpjs-indonesia/Data Faskes BPJS 2019.csv", header=0)
data.head()
data.info()
#Cleaning function

#source: https://www.kaggle.com/anggriyulio/cleaning-data



def cleanNama(row):

    str = row.replace('Kode Faskes dan Alamat Rumah Sakit BPJS di','')

    return str



def removeSpace(row):

    str = " ".join(row.split())

    return str
data['KotaKab'] = data['KotaKab'].apply(removeSpace).apply(cleanNama)

data['NamaFaskes'] = data['NamaFaskes'].apply(removeSpace)

data['TelpFaskes'] = data['TelpFaskes'].apply(removeSpace)

data.drop(['Link', 'NoLink'], axis=1, inplace=True)



data.head()
data['NamaFaskes'] = data['NamaFaskes'].apply(removeSpace)

data['TelpFaskes'] = data['TelpFaskes'].apply(removeSpace)
data.head()
def searchLatLong(row):

    str = re.search('(-?([0-9]{1}|[0-9]0|[1-8]{1,2}).[0-9]{1,6},(-?(1[0-8]{1,2}|9[1-9]{1}).[1-9]{1,6}))', row)

    if str:

        return str.group()

    return np.NaN



data['LatLongFaskes'] = data['LatLongFaskes'].apply(searchLatLong)



lat = []

lon = []



for row in data['LatLongFaskes']:

    try:

        latitude = float(row.split(',')[0])

        longitude = float(row.split(',')[1])

        if (-90.0 <= latitude <= 90.0):

            lat.append(latitude)

        else:

            lat.append(np.NaN)

            

        if (-180 <= longitude <= 180):

            lon.append(longitude)

        else:

            lon.append(np.NaN)      

    except:

        lat.append(np.NaN)

        lon.append(np.NaN)

        

data['Latitude'] = lat

data['Longitude'] = lon



data.drop(['LatLongFaskes'], axis=1, inplace=True)

data.head()
provinsi = data['Provinsi'].unique().tolist()

print(provinsi)

print('\nTotal provinsi: ', len(provinsi))
kabkota = data['KotaKab'].unique().tolist()

print(kabkota)

print('\Total kota: ', len(kabkota))
total_data_count = data.shape[0]

geographically_available_data_count = data.dropna().shape[0]

geographically_unavailable_data_count = data[data['Latitude'].isnull() | data['Longitude'].isnull()].shape[0]



print('\nTotal data fasilitas: ', total_data_count)

print('\nTotal fasilitas yang dapat ditampilkan di map: ', geographically_available_data_count)

print('\nTotal fasilitas yang tidak dapat ditampilkan di map: ', geographically_unavailable_data_count)

data_map = data.dropna()

data_map.info()
data_map = data.dropna()

rome_lat, rome_lng = -6.200000, 106.816666       

# init the folium map object

my_map = folium.Map(location=[rome_lat, rome_lng], zoom_start=5)

# add all the point from the file to the map object using FastMarkerCluster

my_map.add_child(plugins.FastMarkerCluster(data_map[['Latitude', 'Longitude']].values.tolist()))



my_map
data_kotakab = data['KotaKab'].value_counts().rename_axis('KotaKab').reset_index(name='Jumlah')

data_kotakab.head()
# Mengambil latitude dan longitude data per kota/kabupaten

kabkota = []

prov = []

latk = []

lonk = []



data_n = data.dropna() #hapus kota/kab tanpa latitude/longitude agar data dapat dibuat map



for prv, kk, ltk, lnk in zip(data_n['Provinsi'], data_n['KotaKab'], data_n['Latitude'], data_n['Longitude']):

    if (not(kk in kabkota) and not(ltk == np.NaN) and not(lnk == np.NaN)):

        prov.append(prv)

        kabkota.append(kk)

        latk.append(ltk)

        lonk.append(lnk)



print(len(kabkota))



data_map = pd.DataFrame(list(zip(prov, kabkota, latk, lonk)), columns=['Provinsi', 'KotaKab', 'Latitude', 'Langitude'])

data_map.head()

                          
merged = pd.merge(data_map, data_kotakab, on='KotaKab')

merged.count()
# Create map

bpjs_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, prov, kotakab, jml in zip(merged['Latitude'], merged['Langitude'], merged['Provinsi'], merged['KotaKab'], merged['Jumlah']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Provinsi</strong>: ' + str(prov) + '<br>'

                                '<strong>Kota/Kab</strong>: ' + str(kotakab) + '<br>'

                                '<strong>Jumlah</strong>: ' + str(jml)),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.3 ).add_to(bpjs_map)

bpjs_map

data_sorted = pd.DataFrame({'Provinsi':merged['KotaKab'], 'Jumlah':merged['Jumlah']})

data_sorted = data_sorted.sort_values(by=['Jumlah'], ascending=False)



data_sorted.head()
temp_data = pd.DataFrame({'KotaKab':data_kotakab['KotaKab'], 'Jumlah':data_kotakab['Jumlah']})

#data_sorted = temp_data.sort_values(by=['Jumlah'], ascending=False)[:30]



f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("muted")

sns.barplot(x="Jumlah", y="KotaKab", data=data_kotakab[:30],

           label="Jumlah", color="g")



data_kotakab['Jumlah'].sum()
temp_data = pd.DataFrame({'KotaKab':data_kotakab['KotaKab'], 'Jumlah':data_kotakab['Jumlah']})

data_sorted_asc = temp_data.sort_values(by=['Jumlah'], ascending=True)



f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("muted")

sns.barplot(x="Jumlah", y="KotaKab", data=data_kotakab.tail(30),

           label="Jumlah", color="g")

data_provinsi = data['Provinsi'].value_counts().rename_axis('Provinsi').reset_index(name='Jumlah')

data_provinsi.head()
temp_data = pd.DataFrame({'Provinsi':data_provinsi['Provinsi'], 'Jumlah':data_provinsi['Jumlah']})

data_sorted = temp_data.sort_values(by=['Jumlah'], ascending=False)



f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("muted")

sns.barplot(x="Jumlah", y="Provinsi", data=data_provinsi,

           label="Jumlah", color="g")