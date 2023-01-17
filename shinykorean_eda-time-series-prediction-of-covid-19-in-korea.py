import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from statsmodels.tsa.arima_model import ARIMA

from datetime import datetime, timedelta
df_patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

df_route = pd.read_csv('/kaggle/input/coronavirusdataset//route.csv')

df_time = pd.read_csv('/kaggle/input/coronavirusdataset//time.csv')
df_patient.head()
df_patient.isna().sum()
df_patient['birth_year'] = df_patient.birth_year.fillna(0.0).astype(int)

df_patient['birth_year'] = df_patient['birth_year'].map(lambda x: x if x > 0 else np.nan)
df_patient.confirmed_date = pd.to_datetime(df_patient.confirmed_date)
explode = (0, 0.1)  

fig1, ax1 = plt.subplots()

ax1.pie(df_patient.sex.value_counts().values, explode=explode, 

        labels=df_patient.sex.value_counts().index, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.tight_layout()

plt.show()
df_patient['age'] = datetime.now().year - df_patient.birth_year 
plt.figure(figsize=(15, 5))

plt.title('age')

df_patient.age.hist();
df_patient.infection_reason = df_patient.infection_reason.map(lambda x : str(x).strip())
plt.figure(figsize=(15,5))

plt.title('Infection reason')

df_patient[df_patient.infection_reason != 'nan'].infection_reason.value_counts().plot.bar();
plt.figure(figsize=(15,5))

plt.title('Region')

df_patient.region.value_counts().plot.bar();
plt.figure(figsize=(15,5))

plt.title('Number patients in city')

df_route.city.value_counts().plot.bar();
plt.figure(figsize=(15,5))

plt.title('Visit')

df_route.visit.value_counts().plot.bar();
patient_trend_df = df_patient.groupby('confirmed_date').count().iloc[:,[0]]

patient_trend_df.columns = ['confirmed']

patient_cumsum = patient_trend_df.iloc[:,[0]].cumsum()

patient_cumsum.columns = ['confirmed']

plt.figure(figsize=(14,5))

ax = sns.lineplot(x=patient_trend_df.index,y='confirmed', 

             data=patient_trend_df, label='daily patient')

ax.set_ylabel('Daily Count')

ax2 = ax.twinx()

sns.lineplot(x=patient_cumsum.index, y='confirmed', 

             data=patient_cumsum, ax=ax2, color='red', label='Accumulated Patients')

ax.figure.legend()

ax2.set_ylabel('Accumulated Count')

plt.show()
patient_trend_release_df = df_patient.groupby('released_date').count()

patient_release_cumsum = patient_trend_release_df.iloc[:,[0]].cumsum()



patient_trend_decease_df = df_patient.groupby('deceased_date').count()

patient_decease_cumsum = patient_trend_decease_df.iloc[:,[0]].cumsum()
patient_release_cumsum.columns = ['cured']

patient_decease_cumsum.columns = ['death']

patient_accum = pd.merge(patient_release_cumsum,patient_decease_cumsum, left_index=True, right_index=True, how='outer').fillna(method='ffill')
plt.figure(figsize=(14,5))

sns.lineplot(x=patient_accum.index,y='cured', 

            data=patient_accum, label='Accumulated Cured')



sns.lineplot(x=patient_accum.index, y='death', 

             data=patient_accum, color='red', label='Accumulated Death')

plt.xticks(rotation=30)

plt.show()
model = ARIMA(patient_cumsum, order=(1,1,2))

results = model.fit()



results.plot_predict(1, 50)

plt.show()
model = ARIMA(patient_release_cumsum, order=(1,1,0))

results = model.fit()



results.plot_predict(1, 50)

plt.show()
model = ARIMA(patient_decease_cumsum, order=(1,1,0))

results = model.fit()



results.plot_predict(2, 30)

plt.show()