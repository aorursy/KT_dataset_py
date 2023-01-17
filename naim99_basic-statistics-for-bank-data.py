import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm



sns.set()

%matplotlib inline
df = pd.read_csv("../input/bank-marketing/bank.csv", delimiter=";")

df.head()
print("Mean - {}".format(df["age"].mean()))

print("Median - {}".format(df["age"].median()))

print("Min - {}".format(df["age"].min()))

print("Max - {}".format(df["age"].max()))
df.describe()
data = norm.rvs(10.0, 1, size=5000, random_state=0)

sns.distplot(data, kde=False)
data.std()
sns.distplot(df["age"], kde=False) # Show only histogram
sns.distplot(df["age"], bins=50, kde=False)
sns.boxplot(df["age"], orient="v", width=0.2)
sns.regplot(x=df["age"], y=df["balance"], fit_reg=False) # Don't fit a regression line
sns.pairplot(df)
iris = sns.load_dataset("iris")
iris.head()
sns.regplot(iris["petal_width"], iris["petal_length"])
df.corr()
sns.heatmap(df.corr())
sns.countplot(x=df["education"])
sns.barplot(x="age", y="marital", data=df, ci=False) # Don't show confidence interval
sns.factorplot(x="age", y="marital", data=df, hue="loan", col="default", kind="bar", ci=False)