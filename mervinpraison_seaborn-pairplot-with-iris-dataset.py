import seaborn as sns

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

iris=pd.read_csv('../input/Iris.csv')

iris.drop('Id',axis=1,inplace=True)

print(iris.head())

#sns.pairplot(iris)

sns.pairplot(iris, hue="Species", palette="husl", markers=["o", "s", "D"])
