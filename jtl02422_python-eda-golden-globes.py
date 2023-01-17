import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from numpy import mean
data = pd.read_csv('../input/golden-globe-awards/golden_globe_awards.csv')

data.head()
plt.figure(figsize=(15,10))

df = data['nominee'].value_counts().nlargest(10)

sb.barplot(x=df.index, y=df.values)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel("# of Nominations", fontsize=15)

plt.yticks(fontsize=16)

plt.title("10 Most Nominated")
plt.figure(figsize=(15,10))

df = data.loc[data['win']==True]

df = df['nominee'].value_counts().nlargest(10)

sb.barplot(x=df.index, y=df.values)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel("# of Wins", fontsize=15)

plt.yticks(fontsize=16)

plt.title("10 Most Wins")
df = data.loc[data['nominee']=='Jane Fonda']

df['win'].value_counts().plot.pie(autopct='%.1f%%', label="Jane Fonda Wins vs Nominations")

plt.figure(figsize=(18,10))

df = data.loc[data['nominee']=="Meryl Streep"]

sb.countplot(x='film', hue='win', data=df)

plt.xticks(rotation=90, fontsize=12)

plt.xlabel("Film", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.yticks(fontsize=16)

plt.title("Meryl Streep Movies", fontsize=15)
plt.figure(figsize=(15,10))

df = data['year_award'].value_counts()

sb.lineplot(x=df.index, y=df.values)

plt.xticks(fontsize=16)

plt.xlabel("Year", fontsize=15)

plt.ylabel("Nominations", fontsize=15)

plt.yticks(fontsize=16)
plt.figure(figsize=(15,10))

df2 = data.loc[data['win']==True]

df2 = df2['year_award'].value_counts()

sb.lineplot(x=df.index, y=df.values, label="Nominations")

sb.lineplot(x=df2.index, y=df2.values, label="Wins")

plt.xticks(fontsize=16)

plt.xlabel("Year", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.yticks(fontsize=16)

plt.title("Nominations Vs Wins")
plt.figure(figsize=(15,10))

df = data['film'].value_counts().nlargest(10)

df = data[data['film'].isin(df.index)]

sb.countplot(x='film', hue='win', data=df)

plt.xticks(rotation=90, fontsize=16)

plt.xlabel("Film", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.yticks(fontsize=16)

plt.title("Most Nominated Shows/Movies")