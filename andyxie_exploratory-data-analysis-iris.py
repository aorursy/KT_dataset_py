# Libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Set plot style

sns.set(style="white", color_codes=True)



# Read CSV

iris = pd.read_csv("../input/Iris.csv")
# Glimpse the data

iris.head()
iris.info()
sns.pairplot(iris, hue="Species")
sns.lmplot(x='SepalLengthCm', y='PetalLengthCm', hue='Species', data=iris, fit_reg=False,size=10)
g = sns.factorplot(x="Species", y="SepalLengthCm", data=iris, kind="box", size=10, aspect=.7);