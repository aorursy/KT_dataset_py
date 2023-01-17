# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_squared_error



# display settings

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)



# filterning warnings

import warnings

warnings.filterwarnings("ignore")
# reading data from csv and creating dataframe

df = pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")
# displaying first 5 rows

df.head()
# dropping the ID column as it will not be useful in predicting our dependent variable

df.drop(columns="car_ID", inplace=True)
# dimensions of dataframe

print("No. of rows: {}\tNo. of columns: {}".format(*df.shape))
# columns info

df.info()
# descriptive statistics

df.describe().T
# % of missing values

(df.isna().sum() / df.shape[0]) * 100
# converting from numeric to categorical variable type

df["symboling"] = df["symboling"].astype(str)
# extracting make from the values

df["make"] = df['CarName'].str.split(' ', expand=True)[0]
# unique values in make

df["make"].unique()
# correcting the typo errors in make values

df["make"] = df["make"].replace({"maxda":"mazda",

                               "Nissan":"nissan",

                               "porcshce":"porsche",

                               "toyouta":"toyota",

                               "vokswagen":"volkswagen",

                               "vw":"volkswagen"})
# dropping the car name variable

df.drop(columns="CarName", inplace=True)
# categorizing price into standard and high-end

df["price_category"] = df["price"].apply(lambda x: "standard" if x <= 18500 else "high-end")
# creating list of numeric and categorical columns

col_numeric = list(df.select_dtypes(exclude="object"))



col_categorical = list(df.select_dtypes(include="object"))
# visualizing the car make

plt.figure(figsize=(15,6))

df["make"].value_counts().sort_values(ascending=False).plot.bar()

plt.xticks(rotation=90)

plt.xlabel("Make", fontweight="bold")

plt.ylabel("Count", fontweight="bold")

plt.title("Countplot of Car Make", fontweight="bold")

plt.show()
# visualizing the other categorical variables

plt.figure(figsize=(15,20))

for i,col in enumerate(col_categorical[:-2], start=1):

    plt.subplot(5,2,i)

    sns.countplot(df[col])

    plt.xlabel(col, fontweight="bold")

plt.show()
# pair plot to understand the correlation between the numeric variables (except price)

sns.pairplot(df[col_numeric[:-1]])

plt.show()
# heatmap to visualize the pearson's correlation matrix between the numeric variables (except price)

plt.figure(figsize=(12,8))

sns.heatmap(df.drop(columns="price").corr(), annot=True, cmap="RdYlGn", square=True, mask=np.triu(df.drop(columns="price").corr(), k=1))

plt.show()
# visualizing our dependent variable for outliers and skewnwss

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.boxplot(df["price"])

plt.title("Boxplot for outliers detection", fontweight="bold")



plt.subplot(1,2,2)

sns.distplot(df["price"])

plt.title("Distribution plot for skewness", fontweight="bold")



plt.show()
# average price of each make

df.groupby("make")["price"].mean().sort_values(ascending=False).plot.bar(figsize=(12,6))

plt.title("Average price of each make", fontweight="bold")

plt.ylabel("Price", fontweight="bold")

plt.xlabel("Make", fontweight="bold")

plt.show()
# proportion of high-end models in each make

pd.crosstab(df["make"], df["price_category"], normalize="index").plot.bar(stacked=True, figsize=(10,5))

plt.xlabel("Make", fontweight="bold")

plt.ylabel("Proportion", fontweight="bold")

plt.title("Proportion of high-end models in each make", fontweight="bold")

plt.show()
# price analysis for each carbody type

fig, ax = plt.subplots(1,2, figsize=(15,5))



pd.crosstab(df["carbody"], df["price_category"], normalize="index").plot.bar(stacked=True, ax=ax[0])

ax[0].set(xlabel="Carbody type", ylabel="Proportion", title="Proportion of high-end models in each carbody type")



df.groupby("carbody")["price"].mean().sort_values(ascending=False).plot.bar(ax=ax[1])

ax[1].set(xlabel="Carbody type", ylabel="Average price", title="Average price of models in each carbody type")



plt.show()
# visualizing distribution of price with the other categorical variables

plt.figure(figsize=(15,20))

for i,col in enumerate(col_categorical[:-2], start=1):

    plt.subplot(5,2,i)

    sns.violinplot(data=df, x=col, y="price", split=True, hue="price_category")

    plt.xlabel(col, fontweight="bold")

plt.show()
# visualizing distribution of price with continuous variables

col_numeric_pc = col_numeric.copy()

col_numeric_pc.append("price_category")

sns.pairplot(df[col_numeric_pc], hue="price_category")

plt.show()
# heatmap to visualize the pearson's correlation between price and other the numeric variables

plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", square=True, mask=np.triu(df.corr(), k=1))

plt.show()
# converting categorical variables into numeric variables using label encoding

le = LabelEncoder()



df_encoded = df.drop(columns=["price_category"])

df_encoded[col_categorical[:-1]] = df_encoded[col_categorical[:-1]].apply(lambda col: le.fit_transform(col))



df_encoded.head()
# independent variables

X = df_encoded.drop(columns="price")



# dependent variable

y = df_encoded["price"]
# splitting into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# building a base model

base_model = DecisionTreeRegressor()

base_model.fit(X_train, y_train)
# scoring using test data

y_pred = base_model.predict(X_test)

print("R-squared:", r2_score(y_pred, y_test))
# hyperparameter tuning for best model

parameters = {"max_depth":list(range(1,15))}



base_model = DecisionTreeRegressor()

cv_model = GridSearchCV(estimator=base_model, param_grid=parameters, scoring='r2', return_train_score=True, cv=5).fit(X_train,y_train)



pd.DataFrame(cv_model.cv_results_)#[["mean_test_score","mean_train_score"]]



# train and test scores

plt.plot(pd.DataFrame(cv_model.cv_results_)["param_max_depth"], pd.DataFrame(cv_model.cv_results_)["mean_test_score"], label="test score")

plt.plot(pd.DataFrame(cv_model.cv_results_)["param_max_depth"], pd.DataFrame(cv_model.cv_results_)["mean_train_score"], label="train score")

plt.title("Training vs. Test score")

plt.ylabel("R-squared")

plt.xlabel("Max depth")

plt.legend()

plt.grid()

plt.show()
# building final model

model = DecisionTreeRegressor(max_depth=8)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R-squared:", r2_score(y_pred, y_test))