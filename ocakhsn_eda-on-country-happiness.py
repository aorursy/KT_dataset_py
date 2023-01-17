import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
five = pd.read_csv("/kaggle/input/world-happiness-report/2015.csv")
five.head()
sns.distplot(five['Happiness Score'])
plt.figure(figsize=(20, 10))

sns.barplot(data=five.iloc[:20], x="Happiness Score", y="Country")

plt.title("The happiest 20 country")
regions = five.groupby("Region").mean().reset_index()

regions
plt.figure(figsize=(15, 10))

sns.barplot(data=regions, x="Happiness Score", y="Region")

plt.title("Happiness Scores based on Regions")
five[five['Region'] == 'Australia and New Zealand'][['Country', 'Happiness Score', 'Region']]
five[five['Region'] == 'North America'][['Country', 'Happiness Score', 'Region']]
five[five['Region'] == 'Western Europe'][['Country', 'Happiness Score', 'Region']]
five[five['Region'] == 'Sub-Saharan Africa'][['Country', 'Happiness Score', 'Region', 'Economy (GDP per Capita)']]
plt.figure(figsize=(12, 8))

sns.scatterplot(data=five, x="Economy (GDP per Capita)", y="Happiness Score")
regions[['Region', 'Happiness Score', 'Economy (GDP per Capita)']].sort_values(by="Happiness Score", ascending=False)
plt.figure(figsize=(10, 6))

sns.scatterplot(data=five, x="Happiness Score", y="Health (Life Expectancy)")
regions.sort_values(by="Happiness Score", ascending=False)
corr = five.corr()

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(corr,  cmap="YlGnBu", vmax=.3, square=True, annot=True)
five[five['Country'] == "Turkey"]
five.describe()