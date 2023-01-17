%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

full_data = train_data.append(test_data, ignore_index=True)

print ("Number of samples: {} \n Number of variables : {} ."\

       .format(*full_data.shape))


full_data.head()
sb.distplot(full_data['Parch'],kde=False)

plt.show()
sb.distplot(full_data['Age'], hist=False)

plt.show()
plt.figure(figsize=(8,8))

sb.distplot(full_data['Age'])

plt.show()
sb.relplot(x="Age", y="Fare", col="Pclass",

            hue="Sex", style="Sex",

            kind="line", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.scatterplot(x="Age", y="Fare", hue="Sex", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.lineplot(x="Age", y="Fare", hue="Sex", style="Sex", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.barplot(x="Sex", y="Survived", hue="Pclass", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.stripplot(x="Sex", y="Age", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.swarmplot(x="Sex", y="Age", data=full_data)

plt.show()
plt.figure(figsize=(8,8))

sb.boxplot(x="Survived", y="Age", data=full_data)

plt.show()
sb.violinplot(x="Survived", y="Age", hue='Sex', data=full_data)

plt.show()
sb.countplot(x="Survived", data=full_data, palette="Blues");

plt.show()
plt.subplots(figsize=(8, 8))

sb.pointplot(x="Sex", y="Survived", hue="Pclass", data=full_data)

plt.show()
sb.lmplot(x="Age", y="Fare", data=full_data)

plt.show()
plt.subplots(figsize=(10, 10))

sb.regplot(x="Age", y="Fare", data=full_data)

plt.show()
plt.subplots(figsize=(10, 10))

sb.heatmap(full_data.corr(), cmap = "YlGnBu", annot=True, fmt=".2f")

plt.show()
data = full_data[["Pclass", "SibSp", "Parch"]]

survived = full_data["Survived"]

lut = dict(zip(survived.unique(), "rb"))

row_colors = survived.map(lut)



sb.clustermap(data, figsize=(14,12),

              row_colors=row_colors,

              dendrogram_ratio=(.1, .2),

              cbar_pos=(0, .2, .03, .4))
# initialize the FacetGrid object

g = sb.FacetGrid(full_data, col='Survived', row='Pclass')



g.map(plt.hist, 'Age')

g.add_legend()

plt.show()
sb.set_style("ticks")

sb.pairplot(full_data, hue='Sex', diag_kind="hist", kind="scatter", palette="husl")

plt.show()
g = sb.PairGrid(full_data)

g.map(plt.scatter)

plt.show()
g = sb.PairGrid(full_data)

g.map_diag(sb.countplot)

g.map_offdiag(plt.scatter);

plt.show()
g = sb.PairGrid(full_data)

g.map_upper(plt.scatter)

g.map_lower(sb.kdeplot, cmap="Blues_d")

g.map_diag(sb.countplot);

plt.show()
sb.jointplot(x='Age',y='Fare',data=full_data)

plt.show()