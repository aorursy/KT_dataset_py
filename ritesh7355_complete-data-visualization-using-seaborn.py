import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from scipy import stats
df = pd.read_csv(r"../input/upvotesvisualization/train.csv")

df.head()

sns.relplot(x="Views", y="Upvotes", data = df)
sns.relplot(x="Views", y="Upvotes", hue = "Tag", data = df)
sns.relplot(x="Views", y="Upvotes", hue = "Answers", data = df);
sns.relplot(x="Views", y="Upvotes", size = "Tag", data = df);
df2 = pd.read_csv(r"../input/hranalytics/train1.csv")

df2.head()
sns.catplot(x="education", y="avg_training_score", data=df2)

sns.catplot(x="education", y="avg_training_score", jitter = False,  data=df2)

sns.catplot(x="education", y="avg_training_score", hue = "gender", data=df2)

#sns.catplot(x="education", y="avg_training_score", kind = "swarm", data=df2)

sns.catplot(x="education", y="avg_training_score", kind="box", data=df2);
sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind = "box", data=df2)
sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind = "violin", data=df2)
sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind = "violin", split = True, data=df2)
sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind = "bar", data=df2)
sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind = "point", data=df2)
sns.distplot(df2.age)
sns.distplot(df2.age, kde=False, rug = True)
sns.jointplot(x="avg_training_score", y="age", data=df2);
sns.jointplot(x=df2.age, y=df2.avg_training_score, kind="hex", data = df2)
sns.jointplot(x="age", y="avg_training_score", data=df2, kind="kde")
corrmat = df2.corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corrmat, vmax=.8, square=True)
sns.catplot(x="age", y="avg_training_score", data=df2, kind="boxen",height=4, aspect=2.7, hue = "is_promoted")
# Initialize the FacetGrid object

g = sns.FacetGrid(df2, row="gender", hue="gender", aspect=5, height=3)



# # Draw the densities in a few steps

g.map(sns.kdeplot, "age", shade=True, alpha=1, lw=3.5, bw=.2)

g.map(sns.kdeplot, "age", color="w", lw=2, bw=.2)

g.map(plt.axhline, y=0, lw=2)



# # Define and use a simple function to label the plot in axes coordinates

def label(x, color, label):

    ax = plt.gca()

    ax.text(0, .2, label, color=color, ha="left", va="center", transform=ax.transAxes)

g.map(label, "age")



# Set the subplots to overlap

g.fig.subplots_adjust(hspace=-.25)



# # Remove axes details that don't play well with overlap

g.set_titles("")

g.set(yticks=[])

g.despine(bottom=True, left=True)

sns.pairplot(df2)