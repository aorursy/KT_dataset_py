import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from statsmodels.regression.linear_model import OLS

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



zomato = pd.read_csv("../input/zomato.csv", na_values = ["-", ""])

# Making a copy of the data to work on

data = zomato.copy()
data.shape

# The dataset has 51717 rows and 17 columns
data.info()

# Each row represents a restaurant and each column is a feature of the restaurant
data.head(3)
data.tail(3)

data["menu_item"].value_counts()[:1]
data.isnull().sum()
data.address[1]
# Renaming and removing commas in the cost column 

data = data.rename({"approx_cost(for two people)": "cost"}, axis=1)

data["cost"] = data["cost"].replace(",", "", regex = True)
# Converting numeric columns to their appropriate dtypes

data[["votes", "cost"]] = data[["votes", "cost"]].apply(pd.to_numeric)
# Examining restaurant types in the column "listed_in(type)"

data["listed_in(type)"].value_counts()
# Examining the top 20 restaurant types in the column "rest_type"

data["rest_type"].value_counts()[:10]
# Group and aggregate duplicate restaurants that are listed under multiple types in listed_in(type)

grouped = data.groupby(["name", "address"]).agg({"listed_in(type)" : list})

newdata = pd.merge(grouped, data, on = (["name", "address"]))
# Examine the duplicates

newdata.head(3)

# The duplicates can be seen in column "listed_in(type)_x"
# Drop rows which have duplicate information in "name", "address" and "listed_in(type)_x"

newdata["listed_in(type)_x"] = newdata["listed_in(type)_x"].astype(str) # converting unhashable list to a hashable type

newdata.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)

newdata.shape
newdata.describe(include = "all")
# Converting the restaurant names to rownames 

newdata.index = newdata["name"]

# Identifying the top 10 cuisines in Bangalore?

pd.DataFrame(newdata.groupby(["cuisines"])["cuisines"].agg(['count']).sort_values("count", ascending = False)).head(10)

# Dropping unnecessary columns

newdata.drop(["name", "url", "phone", "listed_in(city)", "listed_in(type)_x", "address", "dish_liked",  "listed_in(type)_y", "menu_item", "cuisines", "reviews_list"], axis = 1, inplace = True)

newdata.head(3)
# Converting restaurant ratings to a numeric variable

newdata["rating"] = newdata["rate"].str[:3] # Extracting the first three characters of each string in "rate"

newdata.drop("rate", axis = 1, inplace = True)
# Recreating dataset without NEW restaurants

newdata = newdata[newdata.rating != "NEW"] 
newdata.isnull().sum()
newdata = newdata.dropna(subset = ["rating"])
newdata["rating"] = pd.to_numeric(newdata["rating"])
# Plotting the distribution of restaurant ratings

plt.figure(figsize = (10, 5))

plt.hist(newdata.rating, bins = 20, color = "r")

plt.show()

# Plotting the distribution of locations

plt.figure(figsize = (30, 20))

ax = sns.barplot(data = newdata, x = newdata.location.value_counts().index, y = newdata.location.value_counts())

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") # to make the labels more readable

plt.show()
# Printing restaurant value counts for the top 25 locations

newdata["location"].value_counts()[:25]
# Top 5 locations with the highest ratings

(pd.DataFrame(newdata.groupby("location")["rating"].mean())).sort_values("rating", ascending = False).head(5)

# Top 5 most expensive locations (cost = cost for two)

(pd.DataFrame(newdata.groupby("location")["cost"].mean())).sort_values("cost", ascending = False).head(5)
# Identifying the high rated fancy restaurants on Sankey Road

newdata[(newdata["location"] == "Sankey Road") & (newdata["rating"] >= 4 )]
newdata[(newdata["location"] == "Lavelle Road") & (newdata["rating"] >= 4 )][:10]
# Visualizing the relationship between rating and cost

plt.figure(figsize = (10, 5))

plt.scatter(newdata.rating, newdata.cost)

plt.show()
# Separating the predictors and target

predictors = newdata.drop("rating", axis = 1)

target = newdata["rating"]
# Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(predictors, target, random_state = 0, test_size = 0.3)
# Preprocessing the predictors

num_cols = ["votes", "cost"]

cat_cols = ["location", "rest_type", "online_order", "book_table"]



num_imputer = SimpleImputer(strategy = "median") 

# Imputing numeric columns with the median (not mean because of the high variance)

num_imputed = num_imputer.fit_transform(X_train[num_cols])

scaler = StandardScaler()

# Scaling the numeric columns to have a mean of 0 and standard deviation of 1

num_preprocessed = pd.DataFrame(scaler.fit_transform(num_imputed), columns = num_cols)



cat_imputer = SimpleImputer(strategy = "most_frequent")

# Imputing categorical columns with the mode

cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns = cat_cols)

# Dummifying the categorical columns

cat_preprocessed = pd.DataFrame(pd.get_dummies(cat_imputed, prefix = cat_cols, drop_first = True))

# Joining the numeric and categorical columns and checking their shape

predictors = pd.concat([num_preprocessed, cat_preprocessed], axis=1)
# Dropping the feature with a high VIF 

predictors.drop("rest_type_Quick Bites", axis = 1, inplace = True)

predictors.shape
Y = list(y_train)
# Building an Ordinary Least Squares regression model

import statsmodels.api as sm

X = sm.add_constant(predictors)

ols = sm.OLS(Y, X).fit()
# Predicting on the train data

pred_train = np.around(ols.predict(X), 1)

pred_train[:5] # checking the first 5 predictions
# Preprocessing the test data and predicting on it

test_num_imputed = num_imputer.transform(X_test[num_cols])

test_num_preprocessed = pd.DataFrame(scaler.transform(test_num_imputed), columns = num_cols)



test_cat_imputed = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns = cat_cols)

test_cat_preprocessed = pd.DataFrame(pd.get_dummies(test_cat_imputed, prefix = cat_cols))



test_predictors = pd.concat([test_num_preprocessed, test_cat_preprocessed], axis=1)

test_predictors.drop("rest_type_Quick Bites", axis = 1, inplace = True)



# Accounting for missing columns in the test set caused by dummification

missing_cols = set(predictors) - set(test_predictors)

# Adding missing columns to test set with default value equal to 0

for c in missing_cols:

    test_predictors[c] = 0

# Ensuring the order of column in the test set is in the same order than in train set

test_predictors = test_predictors[predictors.columns]



test_X = sm.add_constant(test_predictors)

test_Y = list(y_train)



# Prediction

pred_test = np.around(ols.predict(test_X), 1)

pred_test[:5] # first five rating predictions
mean_squared_error(y_train, pred_train)
mean_squared_error(y_test, pred_test)
# Finding the Mean Absolute Percentage Error

def mape(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



mape(y_train, pred_train)
mape(y_test, pred_test)
# Printing the model summary

ols.summary()
# Regression with XGBoost

xgb = XGBRegressor(n_estimators = 100, max_depth = 8, gamma = 0.5, colsample_bytree = 0.8, random_state = 0)

xgb.fit(predictors, y_train)



pred_train = xgb.predict(predictors)

pred_test = xgb.predict(test_predictors)
mean_squared_error(y_train, pred_train)
mean_squared_error(y_test, pred_test)
mape(y_train, pred_train)
mape(y_test, pred_test)