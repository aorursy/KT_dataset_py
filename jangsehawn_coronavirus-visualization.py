import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import date, timedelta

from sklearn.cluster import KMeans
path = '/kaggle/input/coronavirusdataset/'

patient_data_path = path + 'patient.csv'

route_data_path = path + 'route.csv'

time_data_path = path + 'time.csv'



patient = pd.read_csv(patient_data_path)

route = pd.read_csv(route_data_path)

time = pd.read_csv(time_data_path)
patient.head()
time.head()
route.head()
patient.isna().sum()
patient['birth_year'] = patient.birth_year.fillna(0.0).astype(int)

patient['birth_year'] = patient['birth_year'].map(lambda val: val if val > 0 else np.nan)
date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    patient[col] = pd.to_datetime(patient[col])
patient["time_to_release_since_confirmed"] = patient["released_date"] - patient["confirmed_date"]

patient["time_to_death_since_confirmed"] = patient["deceased_date"] - patient["confirmed_date"]

patient["duration_since_confirmed"] = patient[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].min(axis=1)

patient["duration_days"] = patient["duration_since_confirmed"].dt.days
patient.confirmed_date = pd.to_datetime(patient.confirmed_date)

daily_count = patient.groupby(patient.confirmed_date).id.count()

accumulated_count = daily_count.cumsum()
accumulated_count.plot()

plt.title('Accumulated Confirmed Count');
patient.released_date = pd.to_datetime(patient.released_date)

daily_count_released = patient.groupby(patient.released_date).id.count()

accumulated_count_released = daily_count_released.cumsum()

accumulated_count_released.plot()

plt.title('Accumulated Relaesed Count');
patient.deceased_date = pd.to_datetime(patient.deceased_date)

daily_count_deceased = patient.groupby(patient.deceased_date).id.count()

accumulated_count_deceased = daily_count_deceased.cumsum()

accumulated_count_deceased.plot()

plt.title('Accumulated deceased Count');
accumulated_count_deceased.plot(label="deceased")

accumulated_count_released.plot(label="released")

plt.legend(loc=2)
accumulated_count_deceased.plot(label="deceased")

accumulated_count_released.plot(label="released")

accumulated_count.plot(label="Confirmed")

plt.legend(loc=2)
patient['age'] = 2020 - patient['birth_year']
dead = patient[patient.state == 'deceased']
dead["age"].describe()

plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the deceased")

sns.kdeplot(data=dead['age'], shade=True)
released = patient[patient.state == 'released']
released['age']. describe()
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the released")

sns.kdeplot(data=released['age'], shade=True)
sns.kdeplot(data=dead['age'],label='deceased', shade=True)

sns.kdeplot(data=released['age'],label='released', shade=True)
male_dead = dead[dead.sex=='male']

female_dead = dead[dead.sex=='female']
plt.figure(figsize=(10,6))

sns.set_style(("ticks"))

plt.title("Age distribution of the deceased by gender")

sns.kdeplot(data=female_dead['age'],label='femele', shade=True)

sns.kdeplot(data=male_dead['age'],label='mele', shade=True)
plt.figure(figsize=(15,5))

plt.title('Infection reason')

patient.infection_reason.value_counts().plot.bar()
plt.figure(figsize=(15,5))

plt.title('Number patients in province')

route.province.value_counts().plot.bar()
plt.figure(figsize=(15,5))

plt.title('Number patients in group')

patient.group.value_counts().plot.bar()
plt.figure(figsize=(15,5))

plt.title('Number patients in infection_reason')

patient.infection_reason.value_counts().plot.bar()
plt.figure(figsize=(25,5))

sns.barplot(

    data= patient,

    x= "age",

    y= "duration_days"

)

plt.title('released by age')