import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from numpy import mean
plt.style.use('ggplot')

data = pd.read_csv('../input/us-accidents/US_Accidents_May19.csv')

print(data.columns)

data.head()
data.dtypes
fig = sb.heatmap(data[['TMC','Severity','Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']].corr(), annot=True, linewidths=0.2)

fig = plt.gcf()

fig.set_size_inches(20,20)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)
plt.figure(figsize=(12,8))

df = data.groupby(['Sunrise_Sunset'])

df.Severity.mean().plot(kind='bar')

plt.ylabel("Severity", fontsize=(15))

plt.xlabel("Sunrise vs Sunset", fontsize=(15))

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.figure(figsize=(20,20))

df = data

df.groupby(['Weather_Condition']).size().nlargest(10).plot.pie(autopct='%1.1f%%', fontsize=(20))
plt.figure(figsize=(15,10))

data['Weather_Condition'].value_counts().nlargest(10).plot.bar()

plt.ylabel("Number of Accidents")

plt.xticks(fontsize=15)
top10 = data['Weather_Condition'].value_counts().nlargest(10)

plt.figure(figsize=(20,10))

df = data[data['Weather_Condition'].isin(top10.index)]

sb.barplot(x='Weather_Condition', y='Severity', estimator=mean, data=df)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel("Weather Conditions", fontsize=15)

plt.ylabel("Severity", fontsize=15)
plt.figure(figsize=(20,10))

sb.barplot(x="Severity", y="Temperature(F)", estimator=mean, data=data, ci=False)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel("Severity", fontsize=20)

plt.ylabel("Temperature", fontsize=20)
plt.figure(figsize=(15,10))

sb.lineplot(x="Severity", y="Visibility(mi)", ci=False, estimator=mean, data=data)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel("Severity", fontsize=20)

plt.ylabel("Visibility(mi)", fontsize=20)