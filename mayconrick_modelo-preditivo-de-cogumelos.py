import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

sns.set(style="darkgrid")

import matplotlib.pyplot as plt # data plotting



import warnings

warnings.simplefilter("ignore") # Remove certain warnings from Machine Learning Models

import pandas as pd

data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

data.head(2)
data.describe()
sns.countplot(data['class'])
fig, ax =plt.subplots(1,3, figsize=(15,5))

sns.countplot(x="cap-shape", hue='class', data=data, ax=ax[0])

sns.countplot(x="cap-surface", hue='class', data=data, ax=ax[1])

sns.countplot(x="cap-color", hue='class', data=data, ax=ax[2])

fig.tight_layout()

fig.show()
fig, ax =plt.subplots(1,2, figsize=(15,5))

sns.countplot(x="bruises", hue='class', data=data, ax=ax[0])

sns.countplot(x="odor", hue='class', data=data, ax=ax[1])

fig.tight_layout()

fig.show()
fig, ax =plt.subplots(1,4, figsize=(20,5))

sns.countplot(x="gill-attachment", hue='class', data=data, ax=ax[0])

sns.countplot(x="gill-spacing", hue='class', data=data, ax=ax[1])

sns.countplot(x="gill-size", hue='class', data=data, ax=ax[2])

sns.countplot(x="gill-color", hue='class', data=data, ax=ax[3])

fig.tight_layout()

fig.show()
fig, ax =plt.subplots(2,3, figsize=(20,10))

sns.countplot(x="stalk-shape", hue='class', data=data, ax=ax[0,0])

sns.countplot(x="stalk-root", hue='class', data=data, ax=ax[0,1])

sns.countplot(x="stalk-surface-above-ring", hue='class', data=data, ax=ax[0,2])

sns.countplot(x="stalk-surface-below-ring", hue='class', data=data, ax=ax[1,0])

sns.countplot(x="stalk-color-above-ring", hue='class', data=data, ax=ax[1,1])

sns.countplot(x="stalk-color-below-ring", hue='class', data=data, ax=ax[1,2])

fig.tight_layout()

fig.show()
fig, ax =plt.subplots(2,2, figsize=(15,10))

sns.countplot(x="veil-type", hue='class', data=data, ax=ax[0,0])

sns.countplot(x="veil-color", hue='class', data=data, ax=ax[0,1])

sns.countplot(x="ring-number", hue='class', data=data, ax=ax[1,0])

sns.countplot(x="ring-type", hue='class', data=data, ax=ax[1,1])

fig.tight_layout()

fig.show()
fig, ax =plt.subplots(1,3, figsize=(20,5))

sns.countplot(x="spore-print-color", hue='class', data=data, ax=ax[0])

sns.countplot(x="population", hue='class', data=data, ax=ax[1])

sns.countplot(x="habitat", hue='class', data=data, ax=ax[2])

fig.tight_layout()

fig.show()
# Make column class True/False for isPoisonous

data['class'].replace('p', 1, inplace = True)

data['class'].replace('e', 0, inplace = True)



# Bruises: t = True / f = False

data['bruises'].replace('t', 1, inplace = True)

data['bruises'].replace('f', 0, inplace = True)
data = pd.get_dummies(data)



pd.set_option("display.max_columns",200)

data.head(5)
Target = ['class']

bruisesColumn = ['bruises']

capColumns = list(data.columns[2:22])

odorColumns = list(data.columns[22:31])

gillColumns = list(data.columns[31:49])

stalkColumns = list(data.columns[49:82])

veilColumns = list(data.columns[82:87])

ringColumns = list(data.columns[87:95])

sporeColumns = list(data.columns[95:104])

populationColumns = list(data.columns[104: 110])

habitatColumns = list(data.columns[110:117])
plt.subplots(figsize=(10,10))

sns.heatmap(data[Target+odorColumns].corr(), annot=True)
plt.subplots(figsize=(10,10))

sns.heatmap(data[Target+populationColumns].corr(), annot=True)
#Create X & y

X = data.iloc[:, 1:]

y = data['class']
#Create Testing and Training Data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn import tree



dtc = tree.DecisionTreeClassifier(max_depth=2, random_state=0)

dtc.fit(X_train, y_train)



dtc.score(X_test, y_test)
import graphviz

dot_data = tree.export_graphviz(dtc, feature_names=X.columns.values, class_names=['Edible', 'Poisonous'], filled=True )

graphviz.Source(dot_data) 