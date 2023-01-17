# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# SETUP

%matplotlib inline

import os

import matplotlib.pyplot as plt

import seaborn as sns

import zipfile

import warnings

warnings.filterwarnings("ignore")
# Load the test and the train tables

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
# Look the train

print("\nTrain structure:\n\n", train.head(5))

# Look the test

print("\nTest structure:\n\n", test.head(5))
print("\nTrain info:\n")

train.info()
print("\nTrain description:\n\n", train.drop('id', axis=1, inplace=False).describe())

print("\nTest description:\n\n", test.drop('id', axis=1, inplace=False).describe())
colors = {

    "Ghost" : "#ff4141",

    "Ghoul" : "#995bbe",

    "Goblin": "#16dc88"

}
sns.set(style="whitegrid", context="talk")

sns.set_color_codes("pastel")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(7, 4))



x, y = [], []

for key in colors:

    x.append(key)

    y.append(train['type'].value_counts()[key])



# Plot the different type occurrences

sns.barplot(x, y, palette=colors)



# Finalize the plot

for n, (label, _y) in enumerate(zip(x, y)):

    ax.annotate( # Attach the counts

        s='{:.0f}'.format(abs(_y)),

        xy=(n, _y),

        ha='center',va='center',

        xytext=(0,10),

        textcoords='offset points',

        weight='bold'

    )

    ax.annotate( # Attach the type label

        s=label,

        xy=(n, _y),

        ha='center',va='center',

        xytext=(0,-15),

        textcoords='offset points',

        weight='bold'

    )



# Add a legend and informative axis label

ax.set(ylabel="Number of occurrences", xlabel="Creature type")

ax.set_xticks([])

plt.setp(f.axes, yticks=[])

plt.title("Monsters occurrences")

plt.tight_layout(h_pad=3)
colormap = {

    "Ghost" : "Reds",

    "Ghoul" : "BuPu",

    "Goblin": "Greens"

}



# Subset the dataset by creature

ghost = train.query("type == 'Ghost'")

ghoul = train.query("type == 'Ghoul'")

goblin = train.query("type == 'Goblin'")



features = list(train.describe().columns[1:])
# Set up the matplotlib figure

f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=False, sharey=False)



ax_ind = 0

for i in range(len(features)-1):

    for j in range(i+1, len(features)):

        # Get the features

        feat1 = features[i]

        feat2 = features[j]

        

        # Set up the figure

        ax = axes.flat[ax_ind]

        ax_ind += 1

        ax.set_aspect("equal")



        # Draw the three density plots

        sns.kdeplot(ghost[feat1], ghost[feat2], ax=ax,

                         cmap=colormap['Ghost'], shade=True, shade_lowest=False, alpha=.6)

        sns.kdeplot(ghoul[feat1], ghoul[feat2], ax=ax,

                         cmap=colormap['Ghoul'], shade=True, shade_lowest=False, alpha=.6)

        sns.kdeplot(goblin[feat1], goblin[feat2], ax=ax,

                         cmap=colormap['Goblin'], shade=True, shade_lowest=False, alpha=.6)



# Conclude

plt.suptitle("Bivariate kernel densities")

f.tight_layout()
# Set up the matplotlib figure

sns.pairplot(train.drop('id', axis=1, inplace=False),

             palette=colors, hue="type",

             diag_kind="kde", diag_kws=dict(shade=True))

plt.suptitle("Pairwise relationships in the dataset")

plt.show()
plt.figure(figsize=(9,4))

sns.countplot(x='color', hue='type', palette=colors, data=train)

plt.suptitle("Distribution of the 'color' class")

plt.show()