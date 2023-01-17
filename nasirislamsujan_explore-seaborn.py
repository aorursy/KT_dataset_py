# loading packages

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings("ignore")
# loading the dataset 

dataset = pd.read_csv("../input/heart.csv")
dataset.head()
dataset.shape
dataset.columns
dataset.describe()
dataset.info()
dataset["age"].unique()
plt.figure(figsize=(16, 5))

sns.set_palette("pastel")

dataset.age.value_counts().plot.bar()
plt.figure(figsize=(16, 5))

sns.countplot(x = "age", hue = "target", data = dataset)

plt.legend(loc="upper right")
facet = sns.FacetGrid(dataset, hue = "target", aspect = 3)

facet.map(sns.kdeplot,"age",shade= True)

facet.set(xlim=(0, dataset["age"].max()))

facet.add_legend()
dataset["sex"].value_counts()
_, ax = plt.subplots(1, 3, figsize=(16, 6))

sns.countplot(x = "sex", hue = "target", data = dataset, ax = ax[0])

sns.swarmplot(x = "sex", y = "age", hue = "target", data = dataset, ax = ax[1])

sns.violinplot(x = "sex", y = "age", hue= "target", split = True, data = dataset, ax=ax[2])

sns.despine(left=True)

plt.legend(loc="upper right")

plt.subplots_adjust(wspace=0.3)
dataset.cp.unique()
_,  ax = plt.subplots(1, 3, figsize=(18, 5))

plt.subplots_adjust(wspace=0.2)

dataset.cp.value_counts().plot.bar(ax = ax[0])

sns.countplot(x = "cp", hue = "target", data = dataset, ax=ax[1])

sns.violinplot(x = "cp", y = "age", hue= "target", split = True, data = dataset, ax = ax[2])

sns.despine(left=True)
_, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.set_palette("deep")

sns.boxplot(x="cp", y="age", hue="target", data=dataset, ax=ax[0])

sns.boxenplot(x = "cp", y = "age", data = dataset, hue="target", ax=ax[1])
dataset["trestbps"].unique()
sns.set_palette("Set2")

plt.figure(figsize=(15, 5))

dataset.trestbps.value_counts().plot.bar()
facet = sns.FacetGrid(dataset, hue = "target", aspect = 3)

facet.map(sns.kdeplot,"trestbps",shade= True)

facet.set(xlim=(0, dataset["trestbps"].max()))

facet.add_legend()
_, ax = plt.subplots(1, 3, figsize=(18, 6))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("RdBu")

sns.boxplot(x = "sex", y = "trestbps", hue="target", data=dataset, ax=ax[0])

sns.boxenplot(x = "sex", y = "trestbps", data = dataset, hue="target", ax=ax[1])

sns.violinplot(x = "sex", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[2])

sns.despine(left=True)
sns.relplot(x = "trestbps", y = "age", hue="target", data=dataset)
dataset.chol.unique()
sns.set_palette("pastel")

facet = sns.FacetGrid(dataset, hue = "target", aspect = 3)

facet.map(sns.kdeplot,"chol",shade= True)

facet.set(xlim=(0, dataset["chol"].max()))

facet.add_legend()
_, ax = plt.subplots(1, 2,figsize=(13, 6))

sns.scatterplot(x = "chol", y = "trestbps", hue= "target", palette = "Reds", data=dataset, ax = ax[0])

sns.scatterplot(x = "chol", y = "age", hue= "target", palette = "deep", data=dataset, ax=ax[1])
dataset.fbs.unique()
dataset.fbs.value_counts()
_, ax = plt.subplots(1, 3, figsize=(18, 6))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("RdYlBu")

sns.boxplot(x = "fbs", y = "trestbps", hue="target", data=dataset, ax=ax[0])

sns.boxenplot(x = "fbs", y = "trestbps", data = dataset, hue="target", ax=ax[1])

sns.violinplot(x = "fbs", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[2])

sns.despine(left=True)
dataset.restecg.unique()
dataset.restecg.value_counts()
dataset.restecg.value_counts().plot.bar()
_, ax = plt.subplots(2, 3, figsize=(18, 12))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("Reds")



sns.boxplot(x = "restecg", y = "age", hue = "target", data = dataset, ax = ax[0][0])

sns.boxenplot(x = "restecg", y = "age", data = dataset, hue = "target", ax = ax[0][1])

sns.violinplot(x = "restecg", y = "age", hue= "target", split = True, data = dataset, ax = ax[0][2])



sns.set_palette("muted")



sns.boxplot(x = "restecg", y = "trestbps", hue="target", data=dataset, ax=ax[1][0])

sns.boxenplot(x = "restecg", y = "trestbps", data = dataset, hue="target", ax=ax[1][1])

sns.violinplot(x = "restecg", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[1][2])

sns.despine(left=True)
dataset.thalach.unique()
_, ax = plt.subplots(1, 2,figsize=(13, 6))

sns.scatterplot(x = "thalach", y = "trestbps", hue= "target", palette = "Reds", data=dataset, ax = ax[0])

sns.scatterplot(x = "thalach", y = "age", hue= "target", palette = "deep", data=dataset, ax=ax[1])
sns.set_palette("pastel")

facet = sns.FacetGrid(dataset, hue = "target", aspect = 3)

facet.map(sns.kdeplot,"thalach",shade= True)

facet.set(xlim=(0, dataset["thalach"].max()))

facet.add_legend()
dataset.exang.unique()
sns.set_palette("deep")

sns.countplot(x = "exang", hue = "target", data = dataset)
_, ax = plt.subplots(2, 3, figsize=(18, 12))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("bone")



sns.boxplot(x = "exang", y = "age", hue = "target", data = dataset, ax = ax[0][0])

sns.boxenplot(x = "exang", y = "age", data = dataset, hue = "target", ax = ax[0][1])

sns.violinplot(x = "exang", y = "age", hue= "target", split = True, data = dataset, ax = ax[0][2])



sns.set_palette("cool")



sns.boxplot(x = "exang", y = "trestbps", hue="target", data=dataset, ax=ax[1][0])

sns.boxenplot(x = "exang", y = "trestbps", data = dataset, hue="target", ax=ax[1][1])

sns.violinplot(x = "exang", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[1][2])

sns.despine(left=True)
dataset.oldpeak.unique()
sns.set_palette("pastel")

facet = sns.FacetGrid(dataset, hue = "target", aspect = 3)

facet.map(sns.kdeplot,"oldpeak",shade= True)

facet.set(xlim=(0, dataset["oldpeak"].max()))

facet.add_legend()
_, ax = plt.subplots(1, 2,figsize=(13, 6))

sns.scatterplot(x = "oldpeak", y = "trestbps", hue= "target", palette = "Set2", data=dataset, ax = ax[0])

sns.scatterplot(x = "oldpeak", y = "age", hue= "target", palette = "deep", data=dataset, ax=ax[1])
dataset.slope.unique()
sns.countplot(x = "slope", hue="target", data = dataset)
_, ax = plt.subplots(3, 3, figsize=(18, 18))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("rocket")



sns.boxplot(x = "slope", y = "age", hue = "target", data = dataset, ax = ax[0][0])

sns.boxenplot(x = "slope", y = "age", data = dataset, hue = "target", ax = ax[0][1])

sns.violinplot(x = "slope", y = "age", hue= "target", split = True, data = dataset, ax = ax[0][2])



sns.set_palette("muted")



sns.boxplot(x = "slope", y = "trestbps", hue="target", data=dataset, ax=ax[1][0])

sns.boxenplot(x = "slope", y = "trestbps", data = dataset, hue="target", ax=ax[1][1])

sns.violinplot(x = "slope", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[1][2])



sns.set_palette("Set2")



sns.boxplot(x = "slope", y = "oldpeak", hue="target", data=dataset, ax=ax[2][0])

sns.boxenplot(x = "slope", y = "oldpeak", data = dataset, hue="target", ax=ax[2][1])

sns.violinplot(x = "slope", y = "oldpeak", hue= "target", split = True, data = dataset, ax = ax[2][2])

sns.despine(left=True)

sns.countplot(x = "ca", hue = "target", data = dataset)
_, ax = plt.subplots(1, 3, figsize=(18, 5))

sns.set_palette("Set2")

sns.boxplot(x = "cp", y = "oldpeak", hue="target", data=dataset, ax=ax[0])

sns.boxenplot(x = "cp", y = "oldpeak", data = dataset, hue="target", ax=ax[1])

sns.violinplot(x = "cp", y = "oldpeak", hue= "target", split = True, data = dataset, ax = ax[2])

sns.despine(left=True)
dataset.thal.unique()
sns.countplot(x = "thal", hue = "target", data = dataset)
_, ax = plt.subplots(2, 3, figsize=(18, 12))

plt.subplots_adjust(wspace=0.3)

sns.set_palette("muted")



sns.boxplot(x = "thal", y = "age", hue = "target", data = dataset, ax = ax[0][0])

sns.boxenplot(x = "thal", y = "age", data = dataset, hue = "target", ax = ax[0][1])

sns.violinplot(x = "thal", y = "age", hue= "target", split = True, data = dataset, ax = ax[0][2])



sns.set_palette("Set2")



sns.boxplot(x = "thal", y = "trestbps", hue="target", data=dataset, ax=ax[1][0])

sns.boxenplot(x = "thal", y = "trestbps", data = dataset, hue="target", ax=ax[1][1])

sns.violinplot(x = "thal", y = "trestbps", hue= "target", split = True, data = dataset, ax = ax[1][2])

sns.despine(left=True)
plt.figure(figsize=(14, 9))

sns.heatmap(data = dataset.corr(), cmap="RdYlBu", fmt=".1f", annot=True, linewidths=1)