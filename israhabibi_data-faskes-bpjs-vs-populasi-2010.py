import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns




data = pd.read_csv('../input/list-faskes-bpjs-indonesia/Data Faskes BPJS 2019.csv', header=0)



def namaKota(row):

    str = row.replace('Kode Faskes dan Alamat Rumah Sakit BPJS di ','')

    return str





def remSpace(row):

    str = " ".join(row.split())

    return str



def searchLatLong(row):

    str = re.search('(-?([0-9]{1}|[0-9]0|[1-8]{1,2}).[0-9]{1,6},(-?(1[0-8]{1,2}|9[1-9]{1}).[1-9]{1,6}))', row)

    if str:

        return str.group()

    return np.NaN



def searchLatLongV2(row):

    str = row.strip('http://maps.google.co.id/?q=').strip('-')

    if str:

        return str

    return np.NaN

        

def valid_latitude(row):

    if float(row) in range(-90.0, 90.0):

        return row

    return np.NaN





def valid_longitude(row):

    return row
data['TelpFaskes'] = data['TelpFaskes'].apply(remSpace)

data['NamaFaskes'] = data['NamaFaskes'].apply(remSpace)

data['KotaKab'] = data['KotaKab'].apply(remSpace)
data['KotaKab'] = data['KotaKab'].apply(namaKota)
data['LatLongFaskes'] = data['LatLongFaskes'].apply(searchLatLongV2)



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

        

        if (-180.0 <= longitude <= 180.0):

            lon.append(longitude)

        else:

            lon.append(np.NaN)

    except:

        lat.append(np.NaN)

        lon.append(np.NaN)



data['Latitude'] = lat

data['Longitude'] = lon

data = data.drop("LatLongFaskes", axis=1)
data = data.astype(str)

data = data.applymap(lambda x: re.sub(r'^-$', str(np.NaN), x))

data.to_csv('Data Faskes BPJS 2019-clean_data.csv')

dataBpjsProv = pd.DataFrame(data.groupby(['Provinsi'])['KodeFaskes'].count())
dataPupulasi = pd.read_csv('../input/pupulasi-indonesia-2010/indo_12_1.csv', header=0, index_col = 0)

dataPupulasi = dataPupulasi[['Pupulasi2010']]
dataPupulasi.head()
dataBpjsProv['JumlahFaskes'] = dataBpjsProv['KodeFaskes']

dataBpjsProv = dataBpjsProv.drop(columns=['KodeFaskes'])

dataBpjsProv.head()
DataFaskesPupulasi = pd.concat([dataBpjsProv,dataPupulasi], axis = 1,sort=True)

DataFaskesPupulasi['RatioPendudukFaskes'] = DataFaskesPupulasi['Pupulasi2010']/DataFaskesPupulasi['JumlahFaskes']
DataFaskesPupulasi.sort_values('RatioPendudukFaskes')