import pandas as pd



data =  pd.read_csv ('USGS_10YearsData.csv')

data.head(5)
data.tail(5)
data.info()
data.isnull().sum()
data.drop('latitude', axis=1, inplace=True)

data.drop('longitude', axis=1, inplace=True)

data.drop('dmin', axis=1, inplace=True)

data.drop('nst', axis=1, inplace=True)

data.drop('gap', axis=1, inplace=True)

data.drop('net', axis=1, inplace=True)

data.drop('id', axis=1, inplace=True)

data.drop('updated', axis=1, inplace=True)

data.drop('horizontalError', axis=1, inplace=True)

data.drop('depthError', axis=1, inplace=True)

data.drop('magError', axis=1, inplace=True)

data.drop('magNst', axis=1, inplace=True)

data.drop('status', axis=1, inplace=True)

data.drop('locationSource', axis=1, inplace=True)

data.drop('magSource', axis=1, inplace=True)

data.drop('rms', axis=1, inplace=True)
data.head(5)
data.info()
# Merubah kolom date menjadi format waktu

data['time'] = pd.to_datetime(data['time'])



# Memisahkan data waktu menjadi kolom waktu yang terpisah

data['Year'] = data.time.dt.year

data['Month'] = data.time.dt.month

data['Day'] = data.time.dt.day

data['Hour'] = data.time.dt.hour

data['Minute'] = data.time.dt.minute

data['Weekday'] = data.time.dt.dayofweek

# Memisahkan data place menjadi kolom Kota dan Negara

data['City'] = data['place'].str.rpartition(',')[0]

data['State'] = data['place'].str.rpartition(',')[2]
data.head(5)
data.drop('time', axis=1, inplace=True)

data.drop('place', axis=1, inplace=True)
data.head()
data.Month.value_counts()
data.City.value_counts()
data.tail(5)
data.type.value_counts()
data.head(5)
data.Year.value_counts()
data.mag.value_counts()
data.depth.value_counts()
data.magType.value_counts()
data.State.value_counts()
import seaborn as sns

import matplotlib.pyplot as plt



year_counts = data.State.value_counts()





plt.figure(figsize=(15,10))

sns.barplot(year_counts.index, year_counts.values)

plt.title('Frekuensi Gempa Tahunan di Seluruh Dunia 1998-2019')

plt.xlabel('Tahun', fontsize=14)

plt.ylabel('Frekuensi Gempa', fontsize=14)

plt.xticks(rotation=90)

plt.show()
from pandas import DataFrame



indonesia = data.loc[data.State == ' Indonesia']

indonesia.head(5)
data.Year.value_counts()
dunia = data.Year.value_counts()





plt.figure(figsize=(15,10))

sns.barplot(dunia.index, dunia.values)

plt.title('Frekuensi Gempa Tahunan di Dunia (2009-2019)', fontsize=30)

plt.xlabel('Tahun', fontsize=24)

plt.ylabel('Frekuensi Gempa', fontsize=24)

plt.show()
indonesia.Year.value_counts()
gempa_indonesia = indonesia.Year.value_counts()





plt.figure(figsize=(15,10))

sns.barplot(gempa_indonesia.index, gempa_indonesia.values)

plt.title('Frekuensi Gempa Tahunan di Indonesia (2009-2019)', fontsize=30)

plt.xlabel('Tahun', fontsize=24)

plt.ylabel('Frekuensi Gempa', fontsize=24)

plt.xticks(rotation=90)

plt.show()
indonesia.mag.mean()
indonesia.mag.median()
gempa1_indonesia = indonesia.mag.value_counts()





plt.figure(figsize=(15,10))

sns.barplot(gempa1_indonesia.index, gempa1_indonesia.values)

plt.title('Frekuensi Gempa Tahunan di Indonesia')

plt.xlabel('magnitudo', fontsize=14)

plt.ylabel('Frekuensi Gempa', fontsize=14)

plt.xticks(rotation=90)

plt.show()
indonesia.type.value_counts()
# most correlated features

corrmat = data.corr()

top_corr_features = corrmat.index[abs(corrmat["mag"])>0.05]

plt.figure(figsize=(10,10))

g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
magnitudo = data.loc[data.mag == 9.1]

magnitudo
nuklir = data.loc[data.type == 'nuclear explosion']

nuklir
vulkanik = data.loc[data.type == 'volcanic eruption']

vulkanik.head(10)
vulkanik2 = vulkanik.City.value_counts()

vulkanik2
plt.figure(figsize=(15,10))

sns.barplot(vulkanik2.index, vulkanik2.values)

plt.title('Gempa Vulkanik berdasarkan Lokasi Titik Gempa (2018)', fontsize=30)

plt.xlabel('Lokasi Gempa Vulkanik', fontsize=24)

plt.ylabel('Frekuensi Gempa', fontsize=24)

plt.xticks(rotation=0)

plt.show()
indonesia.City.value_counts()
palu = indonesia.loc[indonesia.mag == 7.5]

palu