import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])
df_cov.head()
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])
df_cov.head()

china = df_cov['Country/Region'] == 'Mainland China'
df_china = df_cov[china]
ave_china_conf = df_china.Confirmed.mean()
singapore = df_cov['Country/Region'] == 'Singapore'
df_singapore = df_cov[singapore]
ave_singapore_conf = df_singapore.Confirmed.mean()
pd.DafaFrame()
sns.barplot(x = ave_china_conf, y=ave_singapore_conf)

df_cov.isna().sum(axis=1)
df_cov.Suspected.fillna(0)
china = df_cov['Country/Region'] == 'Mainland China'
df_china = df_cov[china]
df_china.head(3)

df_china['Province/State'].unique()
china_hubei = df_china['Province/State'] == 'Hubei'
df_hubei = df_china[china_hubei]
# kenaikan confirmed di provinsi Hubei
sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
sns.lineplot(x='Last Update', y='Confirmed', data=df_hubei)



sns.relplot(x='Last Update', y='Confirmed', data=df_china, hue='Province/State', kind='line', height = 5);
df_cov['Province/State'].unique()
df_cov['Country/Region'].unique()
South_Korea = df_cov['Country/Region'] == 'South Korea'
df_Korea = df_cov[South_Korea]
df_Korea.head(3)
sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
sns.lineplot(x='Last Update', y='Confirmed', data=df_china)

df_cov.describe()
sns.set_style('dark')
sns.relplot(x='Confirmed', y='Recovered', data=df_cov)
sns.relplot(x='Last Update', y='Confirmed', data=df_cov, hue='Province/State', kind='line', height = 5);
plt.figure(figsize=(15, 9))
sns.barplot(x='Country/Region', y='Death', data=df_cov)

plt.title('Average Death')
