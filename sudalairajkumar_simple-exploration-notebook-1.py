# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.cm as cm

from sklearn import manifold

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Number of rows and columns in train data : ",train.shape)

print("Number of rows and columns in test data : ",test.shape)
train.head()
test.head()
# create a pie chart and look at the percentage distribution #

type_series = train.type.value_counts()

labels = (np.array(type_series.index))

sizes = (np.array((type_series / type_series.sum())*100))

colors = ['gold', 'lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Percentage distribution of the 3 classes")

plt.show()



# create a function for labeling #

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')



# create a bar chart to get the count of each classes #

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(type_series), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels)

ax.set_ylabel("Count")

ax.set_title("Count of each of the target classes")

autolabel(rects)

plt.show()
numerical_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']

corrmat = train[numerical_cols].corr()

sns.heatmap(corrmat, vmax=.8, square=True, linewidths=0.05)

plt.yticks(rotation='horizontal')

plt.title("Correlation plot")

plt.show()
plt.figure(figsize=(12,12))

sns.pairplot(train[numerical_cols+['type']], hue="type")

plt.title("Bivariate plot on numerical features")

plt.show()
# create a pie chart and look at the percentage distribution #

type_series = train.color.value_counts()

labels = (np.array(type_series.index))

sizes = (np.array((type_series / type_series.sum())*100))

colors = ['white', 'gold', 'green', 'grey', 'blue', 'red']

plt.figure(figsize=(6,5))

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Percentage distribution of the categorical variable 'color' ")

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='color', hue='type', data=train)

plt.show()
plt.figure(figsize=(12,6))

sns.swarmplot(x="color", y="bone_length", hue="type", data=train)

plt.title("Color Vs BoneLength")

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x="color", y="rotting_flesh", hue="type", data=train)

plt.title("Color Vs Rotting Flesh")

plt.show()
plt.figure(figsize=(12,6))

sns.violinplot(x="color", y="hair_length", hue="type", data=train)

plt.title("Color Vs Hair Length")

plt.show()
plt.figure(figsize=(12,6))

sns.pointplot(x="color", y="has_soul", hue="type", data=train)

plt.title("Color Vs Has Soul")

plt.show()
tsne = manifold.TSNE(n_components=2, random_state=0) 

X = np.array(train[numerical_cols])

X = tsne.fit_transform(X)

y = train.type.values



colors = cm.rainbow(np.linspace(0, 1, 3))

labels = ['Ghoul', 'Goblin', 'Ghost']

plt.figure(figsize=(8, 8))

for l, c, co, in zip(labels, colors, range(3)):

    plt.scatter(X[np.where(y == l), 0],

                X[np.where(y == l), 1],

                marker='o',

                color=c,

                linewidth='1',

                alpha=0.8,

                label=l)

plt.xlabel("Dimension 1")

plt.ylabel("Dimension 2")

plt.title("t-SNE Distribution")

plt.legend(loc='best')

plt.show()