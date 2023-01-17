# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# ignore warnings

import warnings

warnings.filterwarnings('ignore')

fifa19 = pd.read_csv('/kaggle/input/fifa19/data.csv', index_col=0)

fifa19.head()
fifa19.info()
fifa19['Body Type'].value_counts()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

x = pd.Series(x, name="Age variable")

ax = sns.distplot(x, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

ax = sns.distplot(x, bins=10, vertical = True)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

x = pd.Series(x, name="Age variable")

ax = sns.kdeplot(x)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

x = pd.Series(x, name="Age variable")

ax = sns.kdeplot(x, shade=True, color='r')

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

ax = sns.distplot(x, kde=False, rug=True, bins=10)

plt.show()
f, ax = plt.subplots(figsize=(8,6))

x = fifa19['Age']

ax = sns.distplot(x, hist=False, rug=True, bins=10)

plt.show()
fifa19['Preferred Foot'].nunique()
fifa19['Preferred Foot'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Preferred Foot", data=fifa19, color="c")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(x="Preferred Foot", hue="Real Face", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.countplot(y="Preferred Foot", data=fifa19, color="c")

plt.show()
g = sns.catplot(x="Preferred Foot", kind="count", palette="ch:.25", data=fifa19)
fifa19['International Reputation'].nunique()
fifa19['International Reputation'].value_counts()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="International Reputation", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="International Reputation", y="Potential", data=fifa19, jitter=0.01)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 

                   data=fifa19, jitter=0.2, palette="Set2", dodge=True)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", 

                   data=fifa19, palette="Set2", size=20, marker="D",

                   edgecolor="gray", alpha=.25)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x=fifa19["Potential"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="International Reputation", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, palette="Set3")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(x=fifa19["Potential"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(x="International Reputation", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, palette="muted")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", 

               data=fifa19, palette="muted", split=True)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.pointplot(x="International Reputation", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, dodge=True)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", 

              data=fifa19, markers=["o", "x"], linestyles=["-", "--"])

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19)

plt.show()
from numpy import median

f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", data=fifa19, estimator=median)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci=68)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci="sd")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x="International Reputation", y="Potential", data=fifa19, capsize=0.2)

plt.show()
g = sns.relplot(x="Overall", y="Potential", data=fifa19)
f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x="Height", y="Weight", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.lineplot(x="Stamina", y="Strength", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="Overall", y="Potential", data=fifa19)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="Overall", y="Potential", data=fifa19, color= "g", marker="+")

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.regplot(x="International Reputation", y="Potential", data=fifa19, x_jitter=.01)

plt.show()
g= sns.lmplot(x="Overall", y="Potential", data=fifa19)

g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19)
g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19, palette="Set1")
g= sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19)
g = sns.FacetGrid(fifa19, col="Preferred Foot")
g = sns.FacetGrid(fifa19, col="Preferred Foot")

g = g.map(plt.hist, "Potential")
g = sns.FacetGrid(fifa19, col="Preferred Foot")

g = g.map(plt.hist, "Potential", bins=10, color="r")
g = sns.FacetGrid(fifa19, col="Preferred Foot")

g = (g.map(plt.scatter, "Height", "Weight", edgecolor="w").add_legend())
g = sns.FacetGrid(fifa19, col="Preferred Foot", height=5, aspect=1)

g = g.map(plt.hist, "Potential")
fifa19_new = fifa19[['Age', 'Potential', 'Strength', 'Stamina', 'Preferred Foot']]
g = sns.PairGrid(fifa19_new)

g = g.map(plt.scatter)
g = sns.PairGrid(fifa19_new)

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
g = sns.PairGrid(fifa19_new, hue="Preferred Foot")

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()

g = sns.PairGrid(fifa19_new, hue="Preferred Foot")

g = g.map_diag(plt.hist, histtype="step", linewidth=3)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
g = sns.PairGrid(fifa19_new, vars=['Age', 'Stamina'])

g = g.map(plt.scatter)
g = sns.PairGrid(fifa19_new)

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.map_diag(sns.kdeplot, lw=3, legend=False)
g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)

g = g.plot(sns.regplot, sns.distplot)
import matplotlib.pyplot as plt
g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)

g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")

g = g.plot_marginals(sns.distplot, kde=False, color=".5")
g = sns.JointGrid(x="Overall", y="Potential", data=fifa19, space=0)

g = g.plot_joint(sns.kdeplot, cmap="Blues_d")

g = g.plot_marginals(sns.kdeplot, shade=True)
g = sns.JointGrid(x="Overall", y="Potential", data=fifa19, height=5, ratio=2)

g = g.plot_joint(sns.kdeplot, cmap="Reds_d")

g = g.plot_marginals(sns.kdeplot, color="r", shade=True)
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.regplot(x="Overall", y="Potential", data=fifa19);
sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19, col_wrap=2, height=5, aspect=1)
def sinplot(flip=1):

    x = np.linspace(0, 14, 100)

    for i in range(1, 7):

        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()
sns.set()

sinplot()
sns.set_style("whitegrid")

sinplot()
sns.set_style("dark")

sinplot()
sns.set_style("white")

sinplot()
sns.set_style("ticks")

sinplot()