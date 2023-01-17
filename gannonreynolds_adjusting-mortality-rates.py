import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt



pd.set_option('precision', 8)

pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)

pd.set_option('display.width', 90)



csv_files = {}

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        csv_files[filename.replace('.csv', '')] = pd.read_csv(os.path.join(dirname, filename))
time = csv_files['Time']

time['mortality_rate'] = time['deceased'] / time['confirmed']

print(time.tail(10))
patient = csv_files['PatientInfo']

patient = patient[['sex', 'age', 'country', 'confirmed_date', 'deceased_date', 'state']]



patient_deaths = patient[patient['deceased_date'].notnull()].copy()



patient_deaths['deceased_date'] = pd.to_datetime(patient_deaths['deceased_date'])

patient_deaths['confirmed_date'] = pd.to_datetime(patient_deaths['confirmed_date'])

patient_deaths['days_til_deceased'] = patient_deaths['deceased_date'] - patient_deaths['confirmed_date']



print(patient_deaths)
negatives = patient_deaths[patient_deaths['days_til_deceased'] < datetime.timedelta(days=0)]

patient_deaths.loc[negatives.index, 'days_til_deceased'] = datetime.timedelta(days=0)



print(patient_deaths)
days_to_death = patient_deaths['days_til_deceased'].mean()

print(days_to_death)
shifted_deaths = time['deceased']

shifted_deaths.index -= 3

time['shifted_deaths'] = shifted_deaths

time['adjusted_mortality'] = time['shifted_deaths'] / time['confirmed']

time['mortality_difference'] = time['adjusted_mortality'] / time['mortality_rate']

differences = time['mortality_difference'].replace([np.inf], np.nan).dropna()

print(time[['date', 'confirmed', 'deceased', 'mortality_rate', 'shifted_deaths', 'adjusted_mortality', 'mortality_difference']].tail(20))



print('\nAverage difference in mortality rate: ', differences.mean())
plt.figure(figsize=(7, 7))

plt.plot(time['mortality_rate'], label='previous mortality rate')

plt.plot(time['adjusted_mortality'], label='adjusted mortality rate')

plt.legend(loc="upper left")

plt.xlabel('Days')

plt.ylabel('Mortality Rate')

plt.title('Mortality vs Adjusted Mortality')