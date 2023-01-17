# Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

%matplotlib inline
iris = pd.read_csv('../input/Iris.csv')



iris.head()
# データの概要

iris.describe()
# データの種の種類の特定　方法１

pd.value_counts(iris['Species'])

# データの種の種類の特定　方法２

iris.Species.unique()
# データの縦と横の大きさ

iris.shape
# データの詳細

iris.info()
# correlation between それぞれの変数

cr = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']

cr_matrix = iris[cr].corr()

heatmap = sns.heatmap(cr_matrix,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cr,xticklabels=cr,cmap='Accent')



# これにより PetalLength と PetalWidth に関係性があるといえる。 0.96 
#相関関係にある２つのデータのcross tabulation可視化

ct = sns.FacetGrid(iris, hue="Species", size=7) 

ct.map(plt.scatter, "PetalLengthCm",  "PetalWidthCm") 

ct.add_legend()

plt.title("Relationship between length and width of Petal")