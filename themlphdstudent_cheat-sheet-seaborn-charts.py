# Import dependencies



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
olympic = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

healthcare=pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')

covid_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

google_playstore=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

iris=pd.read_csv('../input/iris/Iris.csv')

corona_virus = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

youtube=pd.read_csv('../input/youtube-new/USvideos.csv')

shootings=pd.read_csv('../input/us-police-shootings/shootings.csv')
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title('Sepal Length vs. Sepal Width')
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)

plt.title('Sepal Length vs. Sepal Width')
sns.regplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title('Sepal Length vs. Sepal Width with Regression Fit')
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris, marker='+')

plt.title('Sepal Length vs. Sepal Width')
# using sequencial color scheme

sns.scatterplot(x='Age', y='Height', hue='Sex', data=olympic, palette='Blues')

plt.title('Age vs. Height')
# using diverging color scheme

sns.scatterplot(x='Age', y='Height', hue='Sex', data=olympic, palette='PuOr')

plt.title('Age vs. Height')
# using discrete color scheme

sns.scatterplot(x='Age', y='Height', hue='Sex', data=olympic, palette='Set1')

plt.title('Age vs. Height')
sns.lineplot(x='Age', y='Height', data=olympic)
sns.lineplot(x='Age', y='Height', data=olympic, hue='Sex')
sns.lineplot(x='Age', y='Height', data=olympic, estimator=None)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.distplot(olympic['Age'])
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.distplot(olympic['Age'], bins=20)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.distplot(olympic['Age'], kde=False)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['PetalLengthCm'])
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['PetalLengthCm'], shade=True)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['PetalWidthCm'], vertical=True)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['PetalWidthCm'], vertical=True, shade=True)
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['PetalWidthCm'], shade=True)

sns.kdeplot(iris['PetalLengthCm'], shade=True)

plt.show()
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.kdeplot(iris['SepalLengthCm'], shade=True, color='r')

sns.kdeplot(iris['SepalWidthCm'], shade=True, color='m')

plt.show()
sns.kdeplot(iris['SepalLengthCm'], iris['SepalWidthCm'])

plt.title("Sepal Length vs Sepal Width 2D Density Plot")
sns.kdeplot(iris['SepalLengthCm'], iris['SepalWidthCm'], cmap="Reds", shade=True, bw=.15)
sns.kdeplot(iris['SepalLengthCm'], iris['SepalWidthCm'], cmap="Blues", shade=True, shade_lowest=True, )
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.barplot(x='Species', y='SepalLengthCm', data=iris, palette='magma')
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.barplot(x='SepalLengthCm', y='Species', data=iris, palette='magma', orient='h')
sns.boxplot(y=olympic['Age'])

plt.show()
sns.boxplot(x=olympic['Sex'], y=olympic['Height'])

plt.show()
sns.heatmap(iris.drop(['Id'], axis=1).corr())
sns.violinplot(y=shootings['age'])
sns.violinplot(x=shootings['gender'], y=shootings['age'])
sns.violinplot(x=shootings['age'])
sns.violinplot(y=shootings['gender'], x=shootings['age'])
sns.violinplot(data=iris.iloc[:,1:5])
sns.set_style("darkgrid")

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title("Seaborn Dark Grid Theme")
sns.set_style("dark")

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title("Seaborn Dark Theme")
sns.set_style("whitegrid")

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title("Seaborn White Grid Theme")
sns.set_style("white")

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title("Seaborn White Theme")
sns.set_style("ticks")

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)

plt.title("Seaborn Ticks Theme")