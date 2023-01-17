!pip install fbprophet -q
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from datetime import datetime

from fbprophet import Prophet
# read data

path = '/kaggle/input/coronavirusdataset/'

patient = f'{path}patient.csv'

route = f'{path}route.csv'

time = f'{path}time.csv'

df_patient = pd.read_csv(patient)

df_route = pd.read_csv(route)

df_time = pd.read_csv(time)
# date formatting

df_patient['confirmed_date'] = pd.to_datetime(df_patient['confirmed_date'])

df_patient['released_date'] = pd.to_datetime(df_patient['released_date'])

df_patient['deceased_date'] = pd.to_datetime(df_patient['deceased_date'])

df_route['date'] = pd.to_datetime(df_route['date'])
# age group

df_patient['age'] = 2020 - df_patient['birth_year']

bins= [13,25,30,40,60,120]

labels = ['child','yang','adult','elderly','old']

df_patient['age_group'] = pd.cut(df_patient['age'], bins=bins, labels=labels, right=True)
# add days period confirmed -> date

df_patient['days_released'] = df_patient['confirmed_date'] - df_patient['released_date']

df_patient['days_died'] = df_patient['confirmed_date'] - df_patient['deceased_date']

df_patient
confirmed = df_patient['confirmed_date'].value_counts() #format date 2020-02-06

confirmed_df = pd.DataFrame(confirmed).reset_index()

confirmed_df.columns = ['ds', 'y']

train_confirmed = confirmed_df[:-10]

m = Prophet()

m.fit(train_confirmed)

future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)

m.plot_components(forecast)
m.plot(forecast)
male = df_patient.loc[df_patient['sex'] == 'male']

male = male['age_group'].value_counts()

female = df_patient.loc[df_patient['sex'] == 'female']

female = female['age_group'].value_counts()



plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Sex/Age density")

sns.distplot(male, kde=True, color="b", label="Male")

sns.distplot(female, kde=True, color="r", label="Female")
deceased = df_patient.loc[df_patient['state'] == 'deceased']

plt.figure(figsize=(10,10))

deceased['sex'].value_counts().plot.pie(autopct='%1.1f%%')

plt.title('Deceased/sex')
confirmed_date = df_patient['confirmed_date'].value_counts()

confirmed_date = pd.DataFrame(confirmed_date)

confirmed_date.reset_index(inplace=True)

confirmed_date.columns = ['date', 'count']

confirmed_date.describe()
#Age density

age = df_patient['age']



deceased = df_patient.loc[df_patient['state'] == 'deceased']

released = df_patient.loc[df_patient['state'] == 'released']



age_dead = deceased['age']

age_released = released['age']



plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age density")

sns.kdeplot(data=age, shade=False, label="Age")

sns.kdeplot(data=age_dead, shade=True, label="Age of the dead")

sns.kdeplot(data=age_released, shade=True, label="Age of the released")
# Infection reason

#df_patient['infection_reason'].fillna('unknown',inplace=True)

infection_reason = pd.DataFrame(df_patient['infection_reason'].value_counts())

plt.figure(figsize=(10,10))

plt.pie(infection_reason['infection_reason'], labels=infection_reason.index, autopct='%1.1f%%')

plt.title('Infection reason')
"""

df_patient.isna().sum()

df_patient['state'].value_counts()

deceased = df_patient.loc[df_patient['state'] == 'deceased']

deceased"""

"""

     id пациента (n-й подтвержденный пациент)

     sex пол пациента

     birth_year год рождения пациента

     страна страна пациента

     регион регион пациента

     group коллективная инфекция

     infection_reason - причина заражения

     infection_order порядок заражения

     infected_by идентификатором того, кто заразил пациента

     contact_number количество контактов с людьми

     confirmed_date даты подтверждения

     release_date дата выписки

     deceased_date дата смерти

     state  изолирован / освобожден / умер

"""