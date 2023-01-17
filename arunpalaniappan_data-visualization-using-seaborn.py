import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

df_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df_confirmed = df_confirmed.drop(columns = ['Province/State', 'Lat', 'Long'])

df_confirmed = df_confirmed[df_confirmed['Country/Region'].isin(['US', 'India', 'Brazil'])]

df_confirmed = pd.melt(df_confirmed, id_vars = ['Country/Region'], var_name='date', value_name = 'count')

df_confirmed.rename(columns = {'Country/Region':'country'}, inplace=True)

df_confirmed['date'] = pd.to_datetime(df_confirmed['date'])

df_confirmed.head()
plt.figure(figsize=(10, 6))

sns.lineplot(data = df_confirmed, x='date', y='count', hue='country')

plt.show()