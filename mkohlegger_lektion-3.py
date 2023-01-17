import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/dsia19-california-housing/housing.csv")
data.describe()
data.ocean_proximity.value_counts()
data.corr()
# Fehlende Werte finden



data.isna().any()
# Fehlende Werte finden



data.isna().sum()
data.median_income.plot(

    kind="hist",

    title="Median Income"

)
# Einfaches Histogram



plt.figure(figsize=(7,5))

plt.hist(data.median_income)

plt.title("Median Income")

plt.xlabel("Income")

plt.ylabel("abs. Frequency")

plt.show()
plt.scatter(

    x=data.median_income,

    y=data.median_house_value

)

plt.title("Scatter Plot")

plt.xlabel("median income")

plt.ylabel("median house value")
sns.pairplot(data.dropna())

plt.show()
sns.clustermap(data.corr())

plt.show()
# Platz für euren Code