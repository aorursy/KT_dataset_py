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
import time

import pandas as pd

import numpy as np



# import plotting libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

#from pandas.plotting import scatter_matrix

%matplotlib inline

plt.style.use(['fivethirtyeight'])

mpl.rcParams['lines.linewidth'] = 2

import seaborn as sns

#sns.set(style='white', font_scale=.8)

#sns.set_context('talk')



# import the ML algorithm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression



# pre-processing

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



# import libraries for model validation

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



# import libraries for metrics and reporting

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn import metrics

from sklearn.metrics import classification_report



import warnings

warnings.filterwarnings('ignore')
# Reading the dataset

virus = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])

virus.drop('SNo', axis=1, inplace=True)

virus.head()
# Analysing the dataset

print(virus.info())
# Plotting the graph

worldwide_cases_stats = pd.DataFrame(virus.groupby(['Country/Region']).max()

                                  ['Confirmed']).rename_axis('Country').reset_index().sort_values('Confirmed', ascending=False)

ax = sns.catplot(x='Confirmed', y='Country', data=worldwide_cases_stats.head(10), kind='bar', ci='sd', height=8)

ax.set_xticklabels(rotation=45)

plt.title('World scenario')

plt.show()
# Plotting the graphs

outside_china_stats = worldwide_cases_stats[worldwide_cases_stats.Country!='Mainland China'].sort_values('Confirmed', ascending=False)

sns.catplot(x='Confirmed', y='Country', data=outside_china_stats.head(20), kind='bar', ci='sd', height=8)

plt.title('World scenario excluding Mainland China')

plt.show()
# Plotting the graphs

china_stats = virus[virus['Country/Region']=='Mainland China']

province_stats = pd.DataFrame(china_stats.groupby(['Province/State']).max()

                              [['Confirmed','Recovered','Deaths']]).reset_index().rename_axis()



fig, ax = plt.subplots(figsize=(12,12))

sns.barplot(y='Province/State', x='Confirmed', data=province_stats, color='#FFE17B', ax=ax, label='Confirmed')

sns.barplot(y='Province/State', x='Recovered', data=province_stats, color='#B7EC1E', alpha=0.7, ax=ax, label='Recovered')

sns.barplot(y='Province/State', x='Deaths', data=province_stats, color='#FF5733', alpha=0.9, ax=ax, label='Deaths')

plt.xlabel('Confirmed/Recovered/Deaths')

plt.title('Mainland China statistics')

plt.legend()

plt.show()
# Plotting the graphs

province_stats_exclude_hubei = province_stats[province_stats['Province/State']!='Hubei'].sort_values(['Confirmed'],

                                                                                                     ascending=False)

fig, ax = plt.subplots(figsize=(12,12))

sns.barplot(y='Province/State', x='Confirmed', data=province_stats_exclude_hubei, color='#FFE17B', ax=ax, label='Confirmed')

sns.barplot(y='Province/State', x='Recovered', data=province_stats_exclude_hubei, color='#B7EC1E',alpha=0.7, ax=ax, label='Recovered')

sns.barplot(y='Province/State', x='Deaths', data=province_stats_exclude_hubei, color='#FF5733',alpha=0.9, ax=ax, label='Deaths')

#ax.set_xlim(0,1000)

plt.xlabel('Confirmed/Recovered/Deaths')

plt.title('statistics of Mainland China excluding Hubei')

plt.legend()

plt.show()
# Pie diagram

province_stats_hubei = province_stats[province_stats['Province/State']=='Hubei'].sort_values('Confirmed', ascending=False)

fig, ax = plt.subplots()

ax = plt.pie(np.sum(province_stats_hubei.groupby(['Province/State']).max()), labels=['Confirmed','Recovered','Deaths'], 

       autopct='%.2f%%', explode=(0,0,1), shadow=True, radius=2.5, colors=['#FFC300','#DAF7A6','#FF7269'])

plt.legend(loc='lower left', borderaxespad=16, fancybox=True, shadow=True)

plt.show()
# Reading the time series datasets

confirmed_cases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered_cases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

death_cases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
# Making seperate datasets

conf = [] 

for i in confirmed_cases.columns[4:]:

    conf.append(confirmed_cases[i].sum())

Confirmed_ts = pd.DataFrame(data=conf, index=confirmed_cases.columns[4:], columns=['Confirmed'])

recv = []

for j in recovered_cases.columns[4:]:

    recv.append(recovered_cases[j].sum())

Recovered_ts = pd.DataFrame(data=recv, index=recovered_cases.columns[4:], columns=['Recovered'])

dead = []

for k in death_cases.columns[4:]:

    dead.append(death_cases[k].sum())

Deaths_ts = pd.DataFrame(data=dead, index=death_cases.columns[4:], columns=['Deaths'])



# Plotting the curves

fig, ax = plt.subplots(figsize=(16,6))

pos = list(range(len(Confirmed_ts)))

ax = Confirmed_ts.Confirmed.plot(label='Confirmed', color='orange', marker='o', markersize=5)

ax = Recovered_ts.Recovered.plot(label='Recovered', color='green', marker='o', markersize=5)

ax = Deaths_ts.Deaths.plot(label='Deaths', color='red', marker='o', markersize=5)

plt.title('Comparison of Confirmed cases, Recovered and Deaths', size=15)

ax.set_xticks([p for p in pos])

ax.set_xticklabels(Confirmed_ts.index, rotation=45, size=15)

plt.yticks(size=15)

plt.legend(loc="upper left", ncol=3, fancybox=True, shadow=True, fontsize=15)

plt.show()
# Seperate dataset for Mainland China

confirmed_cases_china = confirmed_cases[confirmed_cases['Country/Region']=='Mainland China']

recovered_cases_china = recovered_cases[recovered_cases['Country/Region']=='Mainland China']

death_cases_china = death_cases[death_cases['Country/Region']=='Mainland China']



# Extracting the details

conf_china = []

for i in confirmed_cases_china.columns[4:]:

    conf_china.append(confirmed_cases_china[i].sum())

recv_china = []

for j in confirmed_cases_china.columns[4:]:

    recv_china.append(recovered_cases_china[j].sum())

dead_china = []

for k in confirmed_cases_china.columns[4:]:

    dead_china.append(death_cases_china[k].sum())

res = zip(conf_china,recv_china,dead_china)

counts_china = pd.DataFrame(data=res, index=confirmed_cases_china.columns[4:], columns=['Confirmed', 'Recovered', 'Deaths']) 



# Plotting the curves

fig, ax = plt.subplots(figsize=(16,6))

pos = list(range(len(counts_china)))

ax = counts_china.Confirmed.plot(label='Confirmed', color='orange', marker='o', markersize=5)

ax = counts_china.Recovered.plot(label='Recovered', color='green', marker='o', markersize=5)

ax = counts_china.Deaths.plot(label='Deaths', color='red', marker='o', markersize=5)

ax.set_xticks([p for p in pos])

ax.set_xticklabels(counts_china.index, rotation=45, fontsize=15)

plt.yticks(size=12)

plt.title('Comparison of Confirmed cases, Recovered and Deaths in China', fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.legend(fancybox=True, ncol=3, fontsize=15, shadow=True)

plt.show()
# seperate dataset for Hubei

confirmed_cases_hubei = confirmed_cases_china[confirmed_cases_china['Province/State']=='Hubei']

recovered_cases_hubei = recovered_cases_china[recovered_cases_china['Province/State']=='Hubei']

death_cases_hubei = death_cases_china[death_cases_china['Province/State']=='Hubei']



# Extracting the details

conf_hubei = []

for i in confirmed_cases_hubei.columns[4:]:

    conf_hubei.append(confirmed_cases_hubei[i].sum())

recv_hubei = []

for j in confirmed_cases_hubei.columns[4:]:

    recv_hubei.append(recovered_cases_hubei[j].sum())

dead_hubei = []

for k in confirmed_cases_hubei.columns[4:]:

    dead_hubei.append(death_cases_hubei[k].sum())

res = zip(conf_hubei,recv_hubei,dead_hubei)

counts_hubei = pd.DataFrame(data=res, index=confirmed_cases_hubei.columns[4:], columns=['Confirmed', 'Recovered', 'Deaths'])   



# Plotting

fig, ax = plt.subplots(figsize=(16,6))

pos = list(range(len(counts_hubei)))

ax = counts_hubei.Confirmed.plot(label='Confirmed', color='orange', marker='o', markersize=5)

ax = counts_hubei.Recovered.plot(label='Recovered', color='green', marker='o', markersize=5)

ax = counts_hubei.Deaths.plot(label='Deaths', color='red', marker='o', markersize=5)

ax.set_xticks([p for p in pos])

ax.set_xticklabels(counts_hubei.index, rotation=45, fontsize=15)

plt.yticks(size=12)

plt.title('Comparison of Confirmed cases, Recovered and Deaths in hubei', fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.legend(fancybox=True, ncol=3, fontsize=15, shadow=True)

plt.show()
# Seperate dataset for the rest of the world

confirmed_cases_row = confirmed_cases[confirmed_cases['Country/Region']!='Mainland China']

recovered_cases_row = recovered_cases[recovered_cases['Country/Region']!='Mainland China']

death_cases_row = death_cases[death_cases['Country/Region']!='Mainland China']



# Extracting the details

conf_row = []

for i in confirmed_cases_row.columns[4:]:

    conf_row.append(confirmed_cases_row[i].sum())

recv_row = []

for j in confirmed_cases_row.columns[4:]:

    recv_row.append(recovered_cases_row[j].sum())

dead_row = []

for k in confirmed_cases_row.columns[4:]:

    dead_row.append(death_cases_row[k].sum())

res = zip(conf_row,recv_row,dead_row)

counts_row = pd.DataFrame(data=res, index=confirmed_cases_row.columns[4:], columns=['Confirmed', 'Recovered', 'Deaths'])   



# Plotting the curves

fig, ax = plt.subplots(figsize=(16,6))

pos = list(range(len(counts_row)))

ax = counts_row.Confirmed.plot(label='Confirmed', color='orange', marker='o', markersize=5)

ax = counts_row.Recovered.plot(label='Recovered', color='green', marker='o', markersize=5)

ax = counts_row.Deaths.plot(label='Deaths', color='red', marker='o', markersize=5)

ax.set_xticks([p for p in pos])

ax.set_xticklabels(counts_row.index, rotation=45, fontsize=15)

plt.yticks(size=12)

plt.title('Comparison of Confirmed cases, Recovered and Deaths in Rest of the World', fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.legend(fancybox=True, ncol=3, fontsize=15, shadow=True)

plt.show()
# Plotting the curves

fig, ax = plt.subplots(figsize=(16,6))

pos = list(range(len(counts_china)))

ax = counts_china.Confirmed.plot(label='Mainland China', color='#C70039', marker='o', markersize=5)

ax = counts_hubei.Confirmed.plot(label='Hubei', color='#FF5733', marker='o', markersize=5)

ax = counts_row.Confirmed.plot(label='Rest of the world', color='#FFC300', marker='o', markersize=5)

ax.set_xticks([p for p in pos])

ax.set_xticklabels(counts_china.index, rotation=45, fontsize=15)

plt.yticks(size=15)

plt.title('Comparison of Confirmed cases between Hubei, Mainland China and the rest of the World', fontsize=15)

plt.xlabel('Date', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.legend(loc='upper left', fontsize=15, fancybox=True, ncol=3, shadow=True)

plt.show()