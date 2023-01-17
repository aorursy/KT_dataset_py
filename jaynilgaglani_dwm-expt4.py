import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/weatherclean.csv")
# df.drop('Unnamed: 0',axis=1,inplace=True)

df.columns
df.head()
f, ax = plt.subplots(figsize=(6, 8))

ax = sns.countplot(x="RainTomorrow", data=df, palette="Set1")

plt.show()
plt.figure(figsize=(15,10))





plt.subplot(2, 2, 1)

fig = df.Rainfall.hist(bins=10)

fig.set_xlabel('Rainfall')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 2)

fig = df.WindGustSpeed.hist(bins=10)

fig.set_xlabel('WindGustSpeed')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 3)

fig = df.WindSpeed9am.hist(bins=10)

fig.set_xlabel('WindSpeed9am')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 4)

fig = df.WindSpeed3pm.hist(bins=10)

fig.set_xlabel('WindSpeed3pm')

fig.set_ylabel('RainTomorrow')
correlation = df.corr()

plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Rain in Australia Dataset')

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white',cmap='viridis')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
plt.figure(figsize=(8,8))

plt.subplot(2,2,1)

sns.countplot(data=df,x='WindGustDir')



plt.subplot(2,2,2)

sns.countplot(data=df,x='WindDir9am')



plt.subplot(2,2,3)

sns.countplot(data=df,x='WindDir3pm')

plt.figure(figsize=(9,8))



sns.FacetGrid(df, hue="RainTomorrow", height=4).map(sns.kdeplot, "Humidity9am").add_legend()

sns.FacetGrid(df, hue="RainTomorrow", height=4).map(sns.kdeplot, "Humidity3pm").add_legend()
sns.catplot(x="RainTomorrow",y="RISK_MM",data=df,kind="bar")
sns.lineplot(x="RainTomorrow",y="Rainfall",data=df)