import pandas as pd # data processing
iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()
wine = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
wine.head()
%matplotlib inline
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'])
ax.set_title('Iris dataset')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
fig, ax = plt.subplots()
for i in range(len(iris['SepalLengthCm'])):
    ax.scatter(iris['SepalLengthCm'][i], iris['SepalWidthCm'][i], color=colors[iris['Species'][i]])
ax.set_title('Iris dataset')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
columns = iris.columns.drop(['Species', 'Id'])
fig, ax = plt.subplots()
for column in columns:
    ax.plot(iris[column])
ax.set_title('Iris Dataset')
ax.legend()
fig, ax = plt.subplots()
ax.hist(wine['points'])
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')
fig, ax = plt.subplots()
data = wine['points'].value_counts()
ax.bar(data.index, data.values)
iris.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', title='Iris Dataset')
iris.drop(['Species', 'Id'], axis=1).plot.line(title='Iris Dataset')
wine['points'].plot.hist()
wine['points'].value_counts().sort_index().plot.bar()
import seaborn as sns
sns.__version__
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
sns.lineplot(data=iris.drop(['Species', 'Id'], axis=1))
sns.distplot(wine['points'], bins=10, kde=False)
sns.countplot(wine['points'])
df = wine[(wine['points']>=95) & (wine['price']<1000)]
sns.boxplot('points', 'price', data=df)
sns.violinplot('points', 'price', data=df)
sns.heatmap(iris.corr(), annot=True)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(iris.corr(), ax=ax, annot=True)
g = sns.FacetGrid(iris, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm')
g = sns.FacetGrid(iris, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm', 'SepalWidthCm')
g = sns.FacetGrid(iris, col='Species')
g = g.map(plt.hist, 'PetalLengthCm', bins=10)
sns.pairplot(iris.drop(['Id'], axis=1), hue='Species')
from pandas.plotting import scatter_matrix

scatter_matrix(iris.drop(['Id', 'Species'], axis=1), diagonal='kde', figsize=(10, 10))