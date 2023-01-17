import numpy as np #numerical python bilimsel hesaplamaları hızlı bir şekilde yapmamızı sağlayan bir matematik kütüphanesidir.

import pandas as pd #veri yapıları ve veri analiz araçları sağlayan açık kaynaklı bir BSD lisanslı kütüphanedir.

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



globalTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')

stateTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')

countryTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

cityTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')

majorCityTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByMajorCity.csv')
globalTemp.info()
stateTemp.info()
countryTemp.info()
cityTemp.info()
majorCityTemp.info()
globalTemp.head()



countryTemp.head()

cityTemp.head()

countryTemp[countryTemp.Country=="Turkey"]
cityTemp[cityTemp.City=="Istanbul"]
istanbul_temp=cityTemp.loc[(cityTemp.City.isin(['Istanbul'])) & (cityTemp.AverageTemperature >24)]

istanbul_temp
turkey_temp=countryTemp.loc[(countryTemp.Country.isin(['Turkey'])) & (countryTemp.AverageTemperature >24)]

turkey_temp
turkey_temp=countryTemp.loc[(countryTemp.Country.isin(['Turkey'])) & (countryTemp.dt >'2000-01-01')]

turkey_temp
ortalama=turkey_temp.loc[536515:536522,['dt','AverageTemperature']]

ortalama
ulkeSicakliklari=countryTemp['Country'].unique()

ulkeSicakliklari
plt.figure(figsize=(10,6))

plt.title("Ortalama Sıcaklık 2013 yılı")

sns.barplot(x=ortalama['dt'], y=ortalama['AverageTemperature'])

plt.xlabel("Tarih")

plt.ylabel("Sıcaklık değeri (santigrat)")
plt.figure(figsize=(12,6))

plt.plot(ortalama.dt,ortalama.AverageTemperature) 

plt.show()
globalTemp = globalTemp[['dt', 'LandAverageTemperature']]



globalTemp['dt'] = pd.to_datetime(globalTemp['dt'])

globalTemp['year'] = globalTemp['dt'].map(lambda x: x.year)

globalTemp['month'] = globalTemp['dt'].map(lambda x: x.month)



def get_season(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'autumn'

    else:

        return 'winter'

    

min_year = globalTemp['year'].min()

max_year = globalTemp['year'].max()

years = range(min_year, max_year + 1)



globalTemp['season'] = globalTemp['month'].apply(get_season)



spring_temps = []

summer_temps = []

autumn_temps = []

winter_temps = []



for year in years:

    curr_years_data = globalTemp[globalTemp['year'] == year]

    spring_temps.append(curr_years_data[curr_years_data['season'] == 'spring']['LandAverageTemperature'].mean())

    summer_temps.append(curr_years_data[curr_years_data['season'] == 'summer']['LandAverageTemperature'].mean())

    autumn_temps.append(curr_years_data[curr_years_data['season'] == 'autumn']['LandAverageTemperature'].mean())

    winter_temps.append(curr_years_data[curr_years_data['season'] == 'winter']['LandAverageTemperature'].mean())

sns.set(style="whitegrid")

sns.set_color_codes("pastel")

f, ax = plt.subplots(figsize=(10, 6))



plt.plot(years, summer_temps, label='Yaz mevsimi ortalama sıcaklığı', color='orange')

plt.plot(years, autumn_temps, label='Sonbahar mevsimi ortalama sıcaklığı', color='r')

plt.plot(years, spring_temps, label='İlkbahar mevsimi ortalama sıcaklığı', color='g')

plt.plot(years, winter_temps, label='Kış mevsimi ortalama sıcaklığı', color='b')



plt.xlim(min_year, max_year)



ax.set_ylabel('Ortalama sıcaklık')

ax.set_xlabel('Yıl')

ax.set_title('Her mevsimdeki ortalama sıcaklık')

legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, borderpad=1, borderaxespad=1)
ist = cityTemp.loc[cityTemp['City'] == 'Istanbul', ['dt','AverageTemperature']]

ist.columns = ['Date','Sıcaklık']

ist['Date'] = pd.to_datetime(ist['Date'])

ist.reset_index(drop=True, inplace=True)

ist.set_index('Date', inplace=True)



ist = ist.loc['1900':'2013-01-01']

ist = ist.asfreq('M', method='bfill')

ist.head()
ist['month'] = ist.index.month

ist['year'] = ist.index.year

pivot = pd.pivot_table(ist, values='Sıcaklık', index='month', columns='year', aggfunc='mean')

yilOrt = pd.pivot_table(ist, values='Sıcaklık', index='year', aggfunc='mean')

yilOrt['10 Yıl'] = yilOrt['Sıcaklık'].rolling(10).mean()

yilOrt[['Sıcaklık','10 Yıl']].plot(figsize=(20,6))

plt.title("İstanbul'un yıllara göre ortalama sıcaklık değerleri")

plt.xlabel('Aylar')

plt.ylabel('Sıcaklık')

plt.xticks([x for x in range(1900,2013,3)])

plt.show()