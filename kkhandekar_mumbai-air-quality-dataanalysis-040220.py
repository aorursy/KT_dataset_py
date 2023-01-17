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
#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

%matplotlib inline

import random

from datetime import datetime
#Loading the dataset

data = pd.read_csv("../input/air-quality-in-mumbai-2015/mumbai-aqi-2015.csv")
data.head()
data.shape
#checking for missing values

print(data.isna().sum())
#importing imputation library

from sklearn.impute import SimpleImputer
imp_mean_so2 = SimpleImputer(missing_values=np.nan,  strategy='mean')

imp_mean_no2 = SimpleImputer(missing_values=np.nan,  strategy='mean')

imp_mean_rspm = SimpleImputer(missing_values=np.nan,  strategy='mean')

imp_mean_spm = SimpleImputer(missing_values=np.nan,  strategy='mean')

imp_mean_so2.fit(data[['so2']])

imp_mean_no2.fit(data[['no2']])

imp_mean_rspm.fit(data[['rspm']])

imp_mean_spm.fit(data[['spm']])
data['so2'] = imp_mean_so2.fit_transform(data[['so2']]).ravel()

data['no2'] = imp_mean_no2.fit_transform(data[['no2']]).ravel()

data['rspm'] = imp_mean_rspm.fit_transform(data[['rspm']]).ravel()

data['spm'] = imp_mean_spm.fit_transform(data[['spm']]).ravel()

#confirming if the imputation has worked

data.isna().sum()
data_clean = data.drop(['sampling_date','state','agency', 'pm2_5','stn_code','location' ], axis=1)
#creating a backup of data_clean dataset

data_clean_backup = data_clean.copy()
data_clean.head()
data_clean['location_monitoring_station'].unique()
locn = ['Worli','Kalbadevi', 'Parel']
data_clean['location_monitoring_station'] = data_clean['location_monitoring_station'].replace('Worli, Mumbai',locn[0])

data_clean['location_monitoring_station'] = data_clean['location_monitoring_station'].replace('Bank of India, Kalbadevi Branch, Kalbadevi, Mumbai',locn[1])

data_clean['location_monitoring_station'] = data_clean['location_monitoring_station'].replace('Parel T.T., BMC Southward Office, DR. Ambedkar Road, Mumbai',locn[2])
#verifying if the replacement has worked

data_clean['location_monitoring_station'].unique()
data_clean['date'] = pd.to_datetime(data_clean.date).dt.strftime('%d-%b-%Y')
data_clean.head()
#backing up the dataset

data_clean_backup = data_clean.copy()
pd.options.display.float_format = '{:.2f}'.format 
data_clean.head()
data_clean.groupby("type").size()
types = ['Industrial Areas','Residential','Residential and Rural']

data_clean['type'] = data_clean['type'].replace('Industrial Area',types[0])

data_clean['type'] = data_clean['type'].replace('Residential and others',types[1])

data_clean['type'] = data_clean['type'].replace('Residential, Rural and other Areas',types[2])
#verifying if the replacement has worked

data_clean.groupby("type").size()
#Taking Backup

data_clean_backup = data_clean.copy()
data_clean['Year'] = pd.to_datetime(data_clean['date']).dt.year

data_clean['Month'] = pd.to_datetime(data_clean['date']).dt.strftime('%b')
data_clean.head()
data_clean = data_clean.rename(columns={'location_monitoring_station':'location'})
#Taking Backup

data_clean_backup = data_clean.copy()
plt.style.use('seaborn-deep')

plt.figure(figsize=(10,10))

plt.grid(True)

x = data_clean['so2']

y = data_clean['no2']

plt.hist([x,y], label=['So2 Distribution','No2 Distribution'])

plt.legend(loc='upper right')

plt.title('SO2 & NO2 Distribution')

plt.show()
plt.style.use('seaborn-deep')

plt.figure(figsize=(10,10))

x = data_clean['spm']

y = data_clean['rspm']

plt.hist([x,y], label=['spm Distribution','rspm Distribution'])

plt.legend(loc='upper right')

plt.title('SPM & RSPM Distribution')

plt.show()
#creating a seperate dataframe with feature columns for correlation matrix

data_clean_feature = pd.DataFrame(data_clean, columns= ['so2','no2','spm','rspm'])

data_clean_feature.head()
corr = data_clean_feature.corr(method='pearson')

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data_clean_feature.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data_clean_feature.columns)

ax.set_yticklabels(data_clean_feature.columns)

plt.show()
sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.swarmplot (x='location', y='rspm', data=data_clean, hue = 'type')

plt.title('RSPM Distribution Across Mumbai')
sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.swarmplot (x='location', y='spm', data=data_clean, hue = 'type')

plt.title('SPM Distribution Across Mumbai')
sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.swarmplot (x='location', y='so2', data=data_clean, hue = 'type')

plt.title('SO2 Distribution Across Mumbai')
sns.set(style="darkgrid")

fig, ax = plt.subplots(figsize=(15,10))

ax = sns.swarmplot (x='location', y='no2', data=data_clean, hue = 'type')

plt.title('NO2 Distribution Across Mumbai')
g = plt.figure(figsize=(20,20))

g = sns.catplot(x="Year", y="rspm", hue="location", data=data_clean, kind="bar", palette="muted")

g.set_titles("RSPM Concentration")

g.despine(left=True)

g.set_ylabels("RSPM")

plt.title("RSPM Increase over the Years per Location")
g = plt.figure(figsize=(20,20))

g = sns.catplot(x="Year", y="spm", hue="location", data=data_clean, kind="bar", palette="muted")

g.set_titles("SPM Concentration")

g.despine(left=True)

g.set_ylabels("SPM")

plt.title("SPM Increase over the Years per Location")
g = plt.figure(figsize=(20,20))

g = sns.catplot(x="Year", y="so2", hue="location", data=data_clean, kind="bar", palette="muted")

g.set_titles("SO2 Concentration")

g.despine(left=True)

g.set_ylabels("SO2")

plt.title("SO2 Increase over the Years per Location")
g = plt.figure(figsize=(20,20))

g = sns.catplot(x="Year", y="no2", hue="location", data=data_clean, kind="bar", palette="muted")

g.set_titles("NO2 Concentration")

g.despine(left=True)

g.set_ylabels("NO2")

plt.title("NO2 Increase over the Years per Location")
sns.relplot(x="Year", y="spm", kind="line", hue="type",data=data_clean)

plt.title("SPM Distribution per Year for all location types")
sns.relplot(x="Year", y="rspm", kind="line", hue="type",data=data_clean)

plt.title("RSPM Distribution per Year for all location types")
sns.relplot(x="Year", y="so2", kind="line", hue="type",data=data_clean)

plt.title("SO2 Distribution per Year for all location types")
sns.relplot(x="Year", y="no2", kind="line", hue="type",data=data_clean)

plt.title("NO2 Distribution per Year for all location types")