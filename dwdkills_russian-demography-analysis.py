# data analysis

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='ticks', rc={'figure.figsize': (15, 10)})
data = pd.read_csv("../input/russian-demography/russian_demography.csv")
data.head()
def detailed_analysis(df):

  obs = df.shape[0]

  types = df.dtypes

  counts = df.apply(lambda x: x.count())

  uniques = df.apply(lambda x: [x.unique()])

  nulls = df.apply(lambda x: x.isnull().sum())

  distincts = df.apply(lambda x: x.unique().shape[0])

  missing_ratio = (df.isnull().sum() / obs) * 100

  skewness = df.skew()

  kurtosis = df.kurt()

  print('Data shape:', df.shape)



  cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']

  details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1, sort=False)

  

  details.columns = cols

  dtypes = details.types.value_counts()

  print('____________________________\nData types:\n', dtypes)

  print('____________________________')

  return details
details = detailed_analysis(data)

details
data.describe()
values = data.nlargest(70, columns='npg').drop_duplicates(subset='region')['npg']

indexes = data.nlargest(70, columns='npg')['region']

years = data.nlargest(70, columns='npg').drop_duplicates(subset='region')['year']



fig = plt.figure(figsize=(30, 15))

plot = sns.barplot(indexes, values, ci=None)



for i in range(len(values)):

    plot.text(i-0.18, 2, values.values[i], fontsize=30, rotation=15)

    plot.text(i-0.2, 25, years.values[i], fontsize=30)
fig = plt.figure(figsize=(30, 15))

features = ['npg', 'migratory_growth', 'birth_rate', 'death_rate']

colors = ['g', 'r', 'b', 'orange']



for x in range(len(features)):

    fig1 = fig.add_subplot(2, 2, x + 1)

    sns.boxenplot(data[features[x]], color=colors[x])

    plt.axvline(data[features[x]].mean(), linestyle = "dashed", color ="black", label ="Mean value for the data")

    plt.legend()
correlations = data.corr()



fig = plt.figure(figsize=(12, 10))

sns.heatmap(correlations, annot=True, cmap='GnBu_r', center=1)
sns.regplot(data.birth_rate, data.npg, color='red')

plt.axvline(data.birth_rate.mean(), linestyle = "dashed", color ="black", label ="Mean value for the birth_rate")

plt.axhline(data.npg.mean(), linestyle = "dashed", color ="blue", label ="Mean value for the npg")



plt.legend()
sns.pairplot(data)
sns.jointplot(data.population, data.death_rate, kind='kde', height=10)