import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy

from seaborn import load_dataset

import seaborn as sns
# Loading the data set

#diamonds = sns.load_dataset("diamonds")

# issues with Kaggle, use their data sets

diamonds = pd.read_csv("../input/diamonds/diamonds.csv")

diamonds.head()
#?diamonds
diamonds.rename(columns = {'x': 'length', 'y': 'width', 'z': 'height'}, inplace = True)

diamonds.head()
diamonds.describe()
diamonds.groupby("cut").mean()
# use vectorised cython functions instead.

diamonds.groupby(["cut", "color"]).mean()
diamonds.groupby("clarity").mean()
# unnamed must be removed, leftover from rownames ie index

#diamonds.apply(lambda x: sum(x.notnull())/len(x) * 100)

# rewrite with transform

#diamonds.dtypes

# I conclude that there is no(as far as I know and have researched) more concise alternative to using apply

#diamonds[diamonds.notnull()].groupby(["cut", "color", "clarity"]).transform("sum")

diamonds.notnull().sum() / len(diamonds) * 100
%matplotlib inline

plt.figure()

plt.hist(x = "price", data = diamonds, color = "indianred", alpha = 0.8)

plt.title("Histogram of Prices", fontsize = 15)

plt.ylabel("Frequency", fontsize = 15)

plt.xlabel("Prices(USD)", fontsize = 15)
from scipy.stats import skew

skew(diamonds["price"])
from IPython.display import display

display(np.mean(diamonds["price"]))

display(np.median(diamonds["price"]))
sns.catplot(x="cut", y="price",kind="bar", data = diamonds)

plt.title("Prices by cut", fontsize = 15)
sorted_data = diamonds.sort_values(by = ["price"] , ascending = False)

#sorted_data.head()

# probably a better way exists that leverages col_order

# This sorts the x-axis in alphabetical order which is a bit less informative

sns.catplot(x = "cut", y="price", data = sorted_data, kind = "bar")

plt.scatter(x = "carat", y = "price", data = diamonds, color = "indianred")

plt.title("Prices by carat", fontsize = 15)

plt.xlabel("Carat", fontsize = 14)

plt.ylabel("Price", fontsize = 14)
diamonds[diamonds["carat"] >=5]
diamonds.query("carat >=5")
sns.boxplot(x="cut", y="price", data = diamonds, palette = "RdBu")

plt.title("Price Distribution by Cut", fontsize = 15, fontweight = "bold")

plt.xlabel("cut",size=14)

plt.ylabel("price", size = 15)
filtered = diamonds.query("price > 10000")



sns.boxplot(x = "cut", y = "price", data = filtered, palette = "RdBu")
facets = sns.FacetGrid(col = "cut", row = "clarity", data = diamonds)

facets.map(plt.hist, "price")

plt.title("Price Distribution Plot by Cut and Clarity")

column_order = sorted(diamonds.clarity.unique())

sns.countplot(x="clarity", data = diamonds.sort_values(["price", "depth"]), order = column_order)
sns.violinplot(x="clarity", y="price", data = diamonds.query("price > 10000").sort_values("price"),

              order = column_order)
diamonds.groupby("clarity").mean()
#diamonds.clarity, order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])

# convert to categorical

#pd.Categorical(diamonds["clarity"], categories= ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"] )

# This is probbaly computationally expensive

categorize_data = pd.CategoricalDtype(categories= ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])

diamonds["clairty"] = pd.Series(diamonds.clarity, dtype = categorize_data)

diamonds.groupby("clairty").mean()
# Ok, seaborn's x axis really needs an upgrade. It is currently in my opinion less flexible.

sns.catplot(x = "clarity", y="price", data =  diamonds, kind = "bar",

          col_order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1","SI2","I1"])
from sklearn import linear_model

from sklearn.model_selection import train_test_split
# Make dependent and predictor variables

dependent_variable = diamonds["price"]

predictors = diamonds.drop("price", axis = 1)

display(dependent_variable.describe())

#display(predictors.columns)

display(predictors.head(5))
# split our data into test/train

x_train, x_test, y_train, y_test = train_test_split(predictors, 

                                                    dependent_variable, test_size = 0.2, random_state = 101)

print(len(x_train),len(x_test), len(y_train), len(y_test) )

from sklearn.preprocessing import LabelEncoder

x_train.dtypes
# apply the encoder on cut, color and clarity

# Instantiate the Encoder

encoder = LabelEncoder()

# select_if(is.object, col)

to_encode = x_train.select_dtypes(exclude = 'float').columns.values

x_train[list(to_encode)] = x_train[to_encode].apply(encoder.fit_transform)

x_test[list(to_encode)] = x_test[to_encode].apply(encoder.fit_transform)

x_train.head()
diamonds[to_encode].astype("category").apply(lambda x: x.cat.codes).head()
# make a model object

regressor = linear_model.LinearRegression()

# Use the regressor to fit our model

regressor.fit(x_train, y_train)

# predict on x_test

predicted_price = regressor.predict(x_test)
display("R Squared value(Train): ", regressor.score(x_train, y_train) * 100)

display("R Squared value(Test): ", regressor.score(x_test, y_test) * 100)
y_hat = predicted_price

# sum squared residuals(sse)

ssr = sum((y_test - y_hat)**2)

# sum squared total

sst = sum((y_test - np.mean(y_test)) **2 )

r_squared = 1-(ssr/sst)

# compute the adjusted r squared

adj_r_squared =    1 - (1-regressor.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)- x_test.shape[1]-1)

print(r_squared, adj_r_squared)
from sklearn import metrics
display("MSE: ", metrics.mean_squared_error(predicted_price, y_test))

display("MAE: ", metrics.mean_absolute_error(predicted_price, y_test))

corrs = diamonds.corr()



corrs
sns.heatmap(data = corrs, square = True, cmap = ["dodgerblue", "lightgreen", "gray"], annot = True)
from sklearn.feature_selection import f_regression
f_value, p_value = f_regression(x_train, y_train)

data = pd.DataFrame([f_value, x_train.columns.values, p_value]).T

data.columns = ["f_value", "predictor", "p_value"]

sorted_data = data.sort_values(by = "p_value", ascending = False)

sorted_data
sorted_data["signif" ] = np.where(sorted_data["p_value"] > 0.05 , "ns", "sf")

sorted_data[np.where(sorted_data["signif"] == "sf", True, False)].drop([10,0])
to_index = list(sorted_data["predictor"].values)

# cannot use drop_dupes for some reason

del to_index[2]

# display correlations with price

corrs[[x for x in to_index if x in corrs.columns]].loc["price"]
# Points clustered together. Might be best to reduce this clustering

# Decided to filter for only highly priced diamonds

# perhaps add some noise(jitter?)



sns.lmplot(x = "carat", y = "price", data = diamonds.query("price >= 10000"), hue = "clarity")