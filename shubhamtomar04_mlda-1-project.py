import os

import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import cv2

import tqdm

from IPython.display import Image

%matplotlib inline
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.info()
sns.countplot(x="target", data=df, palette="bwr")

plt.show()
sns.set(rc={'figure.figsize':(15,5)})

df["age"].plot.hist()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5),color=['#1CA53B','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

ax = sns.heatmap(corr_matrix,

                 annot=True,

                 linewidths=0.5,

                 fmt=".2f",

                 cmap="YlGnBu");

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), 

                                                   title="Correlation with target")
ax = sns.heatmap(df, annot=True)
sns.factorplot('age', kind='count', hue='target', data=df, palette='coolwarm', height=10, aspect=.8)
