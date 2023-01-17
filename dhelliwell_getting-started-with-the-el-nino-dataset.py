import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/elnino.csv")



# Remove extra space in columns

df.columns = [col.strip() for col in df.columns]



df.head()
# Air Temp summary statistics

df['Air Temp'] = pd.to_numeric(df['Air Temp'], errors='coerce')

df['Air Temp'].describe()
# Sea Surface Temp summary statistics

df['Sea Surface Temp'] = pd.to_numeric(df['Sea Surface Temp'], errors='coerce')

df['Sea Surface Temp'].describe()
sns.jointplot(x="Air Temp", y="Sea Surface Temp", data=df, size=7)
# Drop some columns and nans before creating heat map.

df_num = df.drop(['Observation', 'Year', 'Month', 'Day', 'Date'], axis=1)

df_num = df_num.apply(pd.to_numeric, errors='coerce')

df_num = df_num.dropna()



sns.heatmap(df_num.corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.jointplot(x="Zonal Winds", y="Sea Surface Temp", data=df_num, size=7)
sns.jointplot(x="Air Temp", y="Humidity", data=df_num, size=7)