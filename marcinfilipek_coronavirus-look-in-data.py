import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime

import seaborn as sns

import networkx as nx
path = '/kaggle/input/coronavirusdataset/'

patient_data_path = path + 'PatientInfo.csv'

route_data_path = path + 'PatientRoute.csv'

time_data_path = path + 'Time.csv'



df_patient = pd.read_csv(patient_data_path)

df_route = pd.read_csv(route_data_path)

df_time = pd.read_csv(time_data_path)
df_patient.head()
df_patient.info()
df_patient.isna().sum()
df_patient.confirmed_date = pd.to_datetime(df_patient.confirmed_date)

df_patient.released_date = pd.to_datetime(df_patient.released_date)

df_patient.deceased_date = pd.to_datetime(df_patient.deceased_date)
df_patient['time_from_confirmed_to_death'] = df_patient.deceased_date - df_patient.confirmed_date

df_patient['time_from_released_to_death'] = df_patient.released_date - df_patient.confirmed_date

df_patient['age'] = datetime.now().year - df_patient.birth_year 
patient_deceased = df_patient[df_patient.state == 'deceased']

patient_isolated = df_patient[df_patient.state == 'isolated']

patient_released = df_patient[df_patient.state == 'released']
f, ax = plt.subplots(figsize=(15, 5))

sns.countplot(y="sex", data=df_patient, color="c");
male_dead = patient_deceased[patient_deceased.sex=='male']

female_dead = patient_deceased[patient_deceased.sex=='female']

plt.figure(figsize=(15,5))

plt.title("Age distribution of the deceased by gender")

sns.kdeplot(data=female_dead['age'], shade=True);

sns.kdeplot(data=male_dead['age'], shade=True);
f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(df_patient.birth_year, color='c');
f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(df_patient.age, color='c');
df_patient.country.value_counts()
f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(y="infection_case", data=df_patient, color="c");
f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(y="province", data=df_patient, color="c");
df_patient.state.value_counts()
f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(patient_deceased.age, color='c');
f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(patient_isolated.age, color='c');
f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(patient_released.age, color='c');
plt.figure(figsize=(15,5))

plt.title('Daily confirmations')

df_patient.groupby('confirmed_date').patient_id.count().plot();
plt.figure(figsize=(15,5))

plt.title('Confirmed count')

df_patient.groupby('confirmed_date').patient_id.count().cumsum().plot();
data_infected_by = df_patient[df_patient.infected_by.notnull()]



def get_sex_for_patient_id(id):

    result = df_patient[df_patient.patient_id == id].sex.values

    return result[0] if len(result) > 0 else 'none'



def get_country_for_patient_id(id):

    result = df_patient[df_patient.patient_id == id].country.values;

    return result[0] if len(result) > 0 else 'none'
values = data_infected_by[['patient_id', 'infected_by']].values.astype(int)



plt.figure(figsize=(20,15))

plt.title("Infection network for all samples\n blue - Korea, red - China, green - rest")

G1=nx.Graph()

G1.add_edges_from(values)

c_map =  ['c' if get_country_for_patient_id(node) == 'Korea' 

          else 'r' if get_country_for_patient_id(node) == 'China' 

          else 'g'

          for node in G1 ]

# without labels - too long

nx.draw(G1,with_labels=False,node_color=c_map, width=3.0, node_size=300)
infected_network_korea = data_infected_by[data_infected_by.country == 'Korea']

values = infected_network_korea[['patient_id', 'infected_by']].values.astype(int)



plt.figure(figsize=(20,15))

plt.title("Infection network in Korea\n blue - male, red - female, green - no data")

G1=nx.Graph()

G1.add_edges_from(values)

c_map =  ['c' if get_sex_for_patient_id(node) == 'male' 

          else 'r' if get_sex_for_patient_id(node) == 'female' 

          else 'g'

          for node in G1 ]

# without labels - too long

nx.draw(G1,with_labels=False,node_color=c_map)
df_route.head()
df_route.info()
f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(y="city", data=df_route, color="c");
f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(y="province", data=df_route, color="c");
f, ax = plt.subplots(figsize=(15, 5))

sns.countplot(y="visit", data=df_route, color="c");
import folium

southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=8,tiles='Stamen Toner')



for lat, lon in zip(df_route['latitude'], df_route['longitude']):

    folium.CircleMarker([lat, lon],

                        radius=5,

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(southkorea_map)

southkorea_map
df_time.head()
df_time.info()
df_time.describe()