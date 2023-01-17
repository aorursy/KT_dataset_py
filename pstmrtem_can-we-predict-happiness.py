import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn



%pylab inline

pylab.rcParams['figure.figsize'] = (10, 7)



df = pd.read_csv("../input/happiness-and-alcohol-consumption/HappinessAlcoholConsumption.csv", sep=",", engine="python")
print("{} countries".format(len(df)))

df.head()
X = df["HappinessScore"]

plt.hist(X, bins=30)

plt.title("Happiness Score Histogram")

plt.xlabel("Hapiness Score")

plt.ylabel("# of rows")

plt.show()
print("Most happy country:")

max_happiness = df["HappinessScore"].max()

df[df["HappinessScore"] == max_happiness]
print("Least happy country:")

min_happiness = df["HappinessScore"].min()

df[df["HappinessScore"] == min_happiness]
print("Most GDP per capita country:")

max_GDP = df["GDP_PerCapita"].max()

df[df["GDP_PerCapita"] == max_GDP]
print("Most beer per capita country:")

max_beer = df["Beer_PerCapita"].max()

df[df["Beer_PerCapita"] == max_beer]
pylab.rcParams['figure.figsize'] = (7, 6)

sn.heatmap(df.corr(), annot=True)

plt.title("Correlation matrix")

plt.show()
pylab.rcParams['figure.figsize'] = (10, 8)



df_sorted = df.sort_values(by=["Beer_PerCapita"])

X, y = df_sorted["Beer_PerCapita"], df_sorted["HappinessScore"]



plt.subplot(5, 1, 1)

plt.plot(X, y)

plt.xlabel("Beer_PerCapita")

plt.ylabel("HappinessScore")

plt.title("Relation between beer consumption and happiness")





df_sorted = df.sort_values(by=["GDP_PerCapita"])

X, y = df_sorted["GDP_PerCapita"], df_sorted["HappinessScore"]



plt.subplot(5, 1, 3)

plt.plot(X, y)

plt.xlabel("GDP_PerCapita")

plt.ylabel("HappinessScore")

plt.title("Relation between GDP and happiness")





df_sorted = df.sort_values(by=["HDI"])

X, y = df_sorted["HDI"], df_sorted["HappinessScore"]



plt.subplot(5, 1, 5)

plt.plot(X, y)

plt.xlabel("HDI")

plt.ylabel("HappinessScore")

plt.title("Relation between HDI and happiness")



plt.show()
df_group_by = df.groupby(["Region"]).mean()

df_group_by
# Predicting happiness using linear model

from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()



X_columns = ["HDI", "GDP_PerCapita", "Beer_PerCapita", "Wine_PerCapita", "Spirit_PerCapita"]



train = df.sample(frac=0.8,random_state=200)

test = df.drop(train.index)



X_train = train[X_columns]

y_train = train["HappinessScore"]

X_test = test[X_columns]

y_test = test["HappinessScore"]



reg.fit(X_train, y_train)

print("R^2 score:", reg.score(X_test, y_test))
# Predicting happiness using linear model, adding polynomial features

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

reg = linear_model.LinearRegression()

poly = PolynomialFeatures(2)



X_columns = ["HDI", "GDP_PerCapita", "Beer_PerCapita", "Wine_PerCapita", "Spirit_PerCapita"]



train = df.sample(frac=0.8,random_state=200)

test = df.drop(train.index)



X_train = train[X_columns]

y_train = train["HappinessScore"]

X_test = test[X_columns]

y_test = test["HappinessScore"]

X_train = poly.fit_transform(X_train)

X_test = poly.fit_transform(X_test)



reg.fit(X_train, y_train)

print("R^2 score with polynomial features:", reg.score(X_test, y_test))
# Adding region to X_columns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

reg = linear_model.LinearRegression()



df_one_hot = pd.get_dummies(df, prefix="region", columns=["Region"])

X_columns = set(df_one_hot.columns.values) - {"Country", "HappinessScore", "Hemisphere"}



train = df_one_hot.sample(frac=0.8,random_state=200)

test = df_one_hot.drop(train.index)



X_train = train[X_columns]

y_train = train["HappinessScore"]

X_test = test[X_columns]

y_test = test["HappinessScore"]



reg.fit(X_train, y_train)

print("R^2 score with region infos:", reg.score(X_test, y_test))
# Adding region and hemisphere to X_columns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

reg = linear_model.LinearRegression()



df_one_hot = pd.get_dummies(df, prefix="region", columns=["Region"])

df_one_hot["Hemisphere"] = df_one_hot["Hemisphere"].replace("noth", "north") # Correcting some misspells

df_one_hot = pd.get_dummies(df_one_hot, prefix="hemisphere", columns=["Hemisphere"])

X_columns = set(df_one_hot.columns.values) - {"Country", "HappinessScore"}



train = df_one_hot.sample(frac=0.8,random_state=200)

test = df_one_hot.drop(train.index)



X_train = train[X_columns]

y_train = train["HappinessScore"]

X_test = test[X_columns]

y_test = test["HappinessScore"]



reg.fit(X_train, y_train)

print("R^2 score with region and hemisphere infos:", reg.score(X_test, y_test))
# New correlation matrix

df_one_hot = pd.get_dummies(df, prefix="region", columns=["Region"])

select_columns = set(df_one_hot.columns) - {"Country", "HDI", "GDP_PerCapita", "Beer_PerCapita", "Spirit_PerCapita", "Wine_PerCapita"}

df_corr = df_one_hot[select_columns]



pylab.rcParams['figure.figsize'] = (7, 6)

sn.heatmap(df_corr.corr(), annot=True)

plt.title("Correlation matrix")

plt.show()
# Final try, with selected columns, and polynomial features

# columns are selected according to the correlation matrices

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

reg = linear_model.LinearRegression()

poly = PolynomialFeatures(2)



df_one_hot = pd.get_dummies(df, prefix="region", columns=["Region"])

train = df_one_hot.sample(frac=0.8,random_state=200)

test = df_one_hot.drop(train.index)



X_columns = ["HDI", "GDP_PerCapita", "region_Sub-Saharan Africa", "region_Western Europe"]

X_train = train[X_columns]

y_train = train["HappinessScore"]

X_test = test[X_columns]

y_test = test["HappinessScore"]

X_train = poly.fit_transform(X_train)

X_test = poly.fit_transform(X_test)



reg.fit(X_train, y_train)

print("R^2 score with selected polynomial features:", reg.score(X_test, y_test))