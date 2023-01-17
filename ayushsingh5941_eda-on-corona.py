import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random
%matplotlib inline
seed = 42

random.seed(seed)
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head()
data.tail()
data.info()
data['Province/State'] = data['Province/State'].fillna('Unknown')
data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')
data['Last Update'] = pd.to_datetime(data['Last Update'])

data['Last Update'] = data['Last Update'].dt.strftime('%d/%m/%Y')
data.info()
# missing data

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(10))
print([pd.value_counts(data[cols]) for cols in data.columns])
data.describe()
data[data.Confirmed > 500]
data['Country'].value_counts().head(30).plot(kind='barh', figsize=(20, 6))
confirmed_plt = sns.relplot(x='Date', y='Confirmed',  data=data, kind='line',aspect=2, height=10, sort=False)

confirmed_plt.set_xticklabels(rotation=-45)
data['Confirmed_cases'] = data['Confirmed'].apply(lambda x: 'Low' if x < 3 else('Medium' if 3 <= x >=6 else 'High'))
data['Confirmed_cases'].value_counts()
data['Deaths_Status'] = data['Deaths'].apply(lambda x: 'No' if x < 1 else 'Yes')
data['Deaths_Status'].value_counts()
data['Recovered_Status'] = data['Recovered'].apply(lambda x: 'No' if x < 1 else('Medium' if 1 <= x >=7 else 'High'))
data['Recovered_Status'].value_counts()
confirmed_death_reco_plt = sns.relplot(x='Date', y='Confirmed', hue='Deaths_Status', size='Recovered_Status', sizes=(50, 200),

                                       data=data, legend='brief', aspect=2)

confirmed_death_reco_plt.set_xticklabels(rotation=-45)
data_no_hubei = data.drop(data[data['Province/State'] == 'Hubei'].index)
confirmed_death_no_hubei_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status', sizes=(50, 200),

                                       data=data_no_hubei, legend='brief', aspect=2)

confirmed_death_no_hubei_plt.set_xticklabels(rotation=-45)
data_conf_1000 = data.iloc[data[data.Confirmed.between(1000, 2000)].index, :]
confirmed_death_no_hubei_1000_plt = sns.relplot(x='Date', y='Confirmed', hue='Deaths_Status', size='Recovered_Status', row='Country',

                                             data=data_conf_1000, legend='brief', aspect=2, kind='line', sort=False)

confirmed_death_no_hubei_1000_plt.set_xticklabels(rotation=-45)
data_conf_800 = data.iloc[data[data.Confirmed.between(100, 500)].index, :]
confirmed_800_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Country', sizes= (20, 200), row='Deaths_Status',

                                             data=data_conf_800, legend='brief', aspect=2)

confirmed_800_plt.set_xticklabels(rotation=-45)
sns.countplot(x='Confirmed_cases', hue='Recovered_Status', data=data)
sns.countplot(x='Confirmed_cases', hue='Deaths_Status', data=data)
sns.countplot(x='Deaths_Status', hue='Recovered_Status', data=data)
data_china = data.iloc[data[data.Country == 'Mainland China'].index, :]
data_china
print('There are ', len(data_china['Province/State'].value_counts()), 'Districts in China where virus is observed')
data_china['Province/State'].value_counts()
china_date_plt = sns.relplot(x='Date', y='Confirmed', data=data_china, aspect=2.5, kind='line', sort=False)

china_date_plt.set_xticklabels(rotation=-45)
data_china_hubei = data.iloc[data[data['Province/State'] == 'Hubei'].index, :]
china_hubei_plt = sns.relplot(x='Date', y='Confirmed', data=data_china_hubei, aspect=2.5,

                                 kind='line', sort=False)

china_hubei_plt.set_xticklabels(rotation=-45)
china_hubei_plt1 = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status', sizes=(100, 20),

                               data=data_china_hubei, aspect=2.5)

china_hubei_plt1.set_xticklabels(rotation=-45)
data_china_no_hubei = data_china.drop(data_china[data_china['Province/State'] == 'Hubei'].index)
china_no_hubei_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status',

                               row='Province/State', data=data_china_no_hubei, aspect=2)

china_no_hubei_plt.set_xticklabels(rotation=-45)