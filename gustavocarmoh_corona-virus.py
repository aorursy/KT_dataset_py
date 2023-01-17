# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data.head()
data.info()
data['Date'] = data['Date'].apply(pd.to_datetime)

data.drop(['Sno'], axis=1, inplace=True)

#data.set_index('Last Update', inplace=True)

data.head()
countries = data['Country'].unique().tolist()

province_state = data['Province/State'].unique().tolist()



print("\n Paises afetados: ", countries)

print("\n Provincias/Estados afetados: ",province_state)



print('\n Total de paises afetados: ', len(countries))

print("\n Total de provincias afetadas pelo virus: ", len(province_state))



#



data['Country'].replace({'Mainland China':'China'}, inplace=True)

countries = data['Country'].unique().tolist()

#print('\n',countries)

#print("\nTotal countries affected by virus: ",len(countries))
d = data['Date'][-1:].astype('str')

ano = int(d.values[0].split('-')[0])

mes = int(d.values[0].split('-')[1])

dia = int(d.values[0].split('-')[2].split()[0])



from datetime import date



ultimos = data[data['Date'] > pd.Timestamp(date(ano,mes,dia))]

ultimos.head()
paises_com_casos = len(ultimos['Country'].value_counts())



casos = pd.DataFrame(ultimos.groupby('Country')['Confirmed'].sum())

casos['Country'] = casos.index

casos.index = np.arange(1,paises_com_casos + 1)



casos_globais = casos[['Country','Confirmed']]

casos_globais
coordenadas = pd.read_csv('../input/world-coordinates/world_coordinates.csv')



info_coordenadas = pd.merge(coordenadas, casos_globais, on='Country')

info_coordenadas.head()
import folium



world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(info_coordenadas['latitude'], info_coordenadas['longitude'], info_coordenadas['Confirmed'], info_coordenadas['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Pais</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Casos Confirmados</strong>: ' + str(int(value)) + '<br>'), 

                        color='orange',

                        fill_color='orange',

                        fill_opacity=0.7).add_to(world_map)

world_map
print('Casos confirmados no mundo: ', int(ultimos['Confirmed'].sum()))

print('Número de mortes confirmadas: ', int(ultimos['Deaths'].sum()))

print('Número de pacientes curados: ', int(ultimos['Recovered'].sum()))
ultimos.groupby(['Country','Province/State']).sum()
ultimos.groupby('Country')['Deaths'].sum().sort_values(ascending=False)[:5]
import matplotlib.pyplot as plt

import seaborn as sns



china = ultimos[ultimos['Country']=='China']



f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=china[1:],

            label="Confirmados", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=china[1:],

            label="Curados", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
china = ultimos[ultimos['Country']!='China']



f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Country", data=china[1:],

            label="Confirmados", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Country", data=china[1:],

            label="Curados", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 30), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
data['Day'] = data['Date'].apply(lambda x:x.day)



plt.figure(figsize=(16,6))

sns.barplot(x='Day',y='Confirmed',data=data)

plt.show()