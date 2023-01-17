import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])
df_cov.head()

df_cov.describe()
df_cov["date"]= df_cov['Last Update'].dt.date.astype('object')

plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Confirmed", data=df_cov)
plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Suspected", data=df_cov)
plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Recovered", data=df_cov)
plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Death", data=df_cov)
plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Confirmed", data=df_cov)
sns.lineplot(x="date", y="Suspected", data=df_cov)
sns.lineplot(x="date", y="Recovered", data=df_cov)
sns.lineplot(x="date", y="Death", data=df_cov)
plt.ylabel("Banyak Orang")

df_cov["Country/Region"].unique()
plt.figure(figsize=(10,10))
sns.barplot(x="Confirmed", y="Country/Region", data=df_cov)

df_italy = df_cov[df_cov["Country/Region"] == "Italy"]
df_italy.head(10)
plt.figure(figsize=(20, 9))
sns.lineplot(x="date", y="Confirmed", data=df_italy)
sns.lineplot(x="date", y="Suspected", data=df_italy)
sns.lineplot(x="date", y="Recovered", data=df_italy)
sns.lineplot(x="date", y="Death", data=df_italy)
plt.ylabel("Banyak Orang")
df_italy.columns
df_italy_state =  df_italy[df_italy["Province/State"].isnull()]
