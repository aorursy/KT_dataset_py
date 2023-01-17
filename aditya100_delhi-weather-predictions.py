import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/testset.csv",parse_dates=['datetime_utc'],skipinitialspace=True)
df.head()
df.describe()
df.shape
plt.figure(figsize=(20, 10))
p = sns.heatmap(df.corr(), annot=True)
df['Date'] = pd.to_datetime(df['datetime_utc'])
df['Year'] = df['Date'].dt.year
p = sns.lineplot(x="Year", y="_dewptm", data=df)
_ = plt.ylabel("Dew")
p = sns.lineplot(x="Year", y="_fog", data=df)
_ = plt.ylabel("Fog")
p = sns.lineplot(x="Year", y="_hum", data=df)
_ = plt.ylabel("Humidity")
p = sns.lineplot(x="Year", y="_heatindexm", data=df)
_ = plt.ylabel("Heat")
p = sns.lineplot(x="Year", y="_rain", data=df)
_ = plt.ylabel("Rain")
plt.figure(figsize=(20, 10))
p = sns.countplot(x='_conds', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)