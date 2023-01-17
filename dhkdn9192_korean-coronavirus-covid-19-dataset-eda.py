import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
patient_df = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

# time_df = pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')

# route_df = pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')



print(f'patient_df.shape: {patient_df.shape}')

# print(f'time_df.shape: {time_df.shape}')

# print(f'route_df.shape: {route_df.shape}')
patient_df.head()
patient_df.info()
# missing values

patient_df.isnull().sum()
confirmed_patients_series = patient_df['confirmed_date'].value_counts().sort_index()



confirmed_patients_series.cumsum().plot(legend='accumulated')

confirmed_patients_series.plot(kind='bar', color='orange', legend='daily', figsize=(16, 5), grid=True)
# before no.31 patient confirmed

limit_series = confirmed_patients_series[:patient_df[patient_df['id'] == 31]['confirmed_date'].values[0]]

limit_series.cumsum().plot(legend='accumulated')

limit_series.plot(kind='bar', color='orange', legend='daily', figsize=(16, 5), grid=True)
# after no.31 patient confirmed

limit_series = confirmed_patients_series[patient_df[patient_df['id'] == 31]['confirmed_date'].values[0]:]

limit_series.cumsum().plot(legend='accumulated')

limit_series.plot(kind='bar', color='orange', legend='daily', figsize=(16, 5), grid=True)
print(patient_df['sex'].value_counts())

sns.countplot(x='sex', data=patient_df)