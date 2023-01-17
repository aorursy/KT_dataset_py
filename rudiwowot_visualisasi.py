import pandas as pd

df = pd.read_csv("../input/covid19-indonesia/covid_19_indonesia_time_series_all.csv")
df
df.info()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.heatmap(df.isnull(), cbar=False)

plt.show()
df['bulan'] = [i.split('/')[0] for i in df['Date']] 

new_df = df[['bulan', 'Total Cases']].groupby('bulan').sum()
new_df

ax = sns.lineplot(new_df.index, new_df['Total Cases'])
from datetime import datetime

df['nama_bulan'] = [datetime.strptime(i, '%m/%d/%Y').strftime("%B") for i in df['Date']]

new_df = df[['nama_bulan', 'Total Cases']].groupby('nama_bulan', sort=False).sum()

plt.figure(figsize=(10,10))
ax = sns.lineplot(x=new_df.index, y=new_df['Total Cases'], sort=False)
plt.xlabel('bulan')