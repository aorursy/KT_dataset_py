



import numpy as np 

import pandas as pd 

from pandas.api.types import is_numeric_dtype

from pandas.api.types import is_string_dtype

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import seaborn as sns

import statsmodels.api as sm





import os

house_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



house_df.shape



house_df.head
pd.set_option("display.max_rows", 81)

all_null = house_df.isnull().sum()

all_null = pd.DataFrame(all_null)

all_null
all_null.rename(columns={all_null.columns[0]: "amount"}, inplace=True)

all_null = all_null[all_null.amount != 0]

print(all_null)
# Three ways to deal with missing values



# Use the average number

# house_df["LotFrontage"] = house_df["LotFrontage"].fillna(house_df["LotFrontage"].mean())



# Use the most common string

# house_df["MSZoning"] = house_df["MSZoning"].fillna(house_df["MSZoning"].mode())



# Delete the whole feature

# house_df.drop(["Alley"], axis=1, inplace=True)

# house_df.shape



#If null values are greater than 30% of total, drop it

too_many_null_features = []

for index, row in all_null.iterrows():

    # print(row.name)

    # print(row["amount"])

    # print(house_df.shape[0])

    if row["amount"] > house_df.shape[0]/3:

        # print(row.name)

        too_many_null_features.append(row.name)

# print(too_big)

for hopeless_feature in too_many_null_features:

    house_df.drop([hopeless_feature], axis=1, inplace=True)

    # drop all the values that have been removed from the dataframe from the null value list

    all_null = all_null.drop(hopeless_feature)

# print(house_df.shape)

print(all_null)







#If remaining values are numbers use the mean

incomplete_nominal_features = []

for index, row in all_null.iterrows():

    if is_numeric_dtype(house_df[row.name][0]):

        print(row.name)

        house_df[row.name] = house_df[row.name].fillna(house_df[row.name].mean())

for incomplete_nominal_feature in incomplete_nominal_features:

    all_null = all_null.drop(incomplete_nominal_feature)

        



#All remaining values use the mode

incomplete_categorical_features = []

for index, row in all_null.iterrows():

    print(row.name)

    house_df[row.name] = house_df[row.name].fillna(house_df[row.name].mode()[0])

for incomplete_categorical_feature in incomplete_categorical_features:

    all_null = all_null.drop(incomplete_categorical_feature)

        
all_null = house_df.isnull().sum()

all_null
# trying a linear regression

linear_x = house_df[["LotFrontage", "LotArea", "OverallQual", "OverallCond"]]

linear_y = house_df["SalePrice"]



linear_model = sm.OLS(linear_y, linear_x).fit()

linear_predictions = linear_model.predict(linear_x)

linear_model.summary()



# RESULTS

# adj R-squared 0.94

# All features are important
# Linear regression with all nominal features

just_nominal_features = []

for number, feature, in enumerate(list(house_df.columns.values)):

    if number > 0:

        if is_numeric_dtype(house_df[feature]) and feature != "SalePrice":

            print(feature)

            just_nominal_features.append(feature)

everything_linear_x = house_df[just_nominal_features]

everything_linear_y = house_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(everything_linear_x, everything_linear_y, test_size = 0.2, random_state = 0)

everything_linear_model = LinearRegression().fit(X_train, y_train)

# everything_linear_predictions = everything_linear_model.predict(X_test)

everything_linear_score = everything_linear_model.score(X_test, y_test)

print("The total score is: " + str(everything_linear_score))
# Prepare for submission

house_test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# Clean Data

all_null = house_test_df.isnull().sum()

all_null = pd.DataFrame(all_null)

all_null.rename(columns={all_null.columns[0]: "amount"}, inplace=True)

all_null = all_null[all_null.amount != 0]



#If null values are greater than 30% of total, drop it

too_many_null_features = []

for index, row in all_null.iterrows():

    if row["amount"] > house_test_df.shape[0]/3:

        too_many_null_features.append(row.name)

for hopeless_feature in too_many_null_features:

    house_test_df.drop([hopeless_feature], axis=1, inplace=True)

    all_null = all_null.drop(hopeless_feature)

#If remaining values are numbers use the mean

incomplete_nominal_features = []

for index, row in all_null.iterrows():

    if is_numeric_dtype(house_test_df[row.name][0]):

        house_test_df[row.name] = house_test_df[row.name].fillna(house_test_df[row.name].mean())

for incomplete_nominal_feature in incomplete_nominal_features:

    all_null = all_null.drop(incomplete_nominal_feature)

#All remaining values use the mode

incomplete_categorical_features = []

for index, row in all_null.iterrows():

    house_test_df[row.name] = house_test_df[row.name].fillna(house_test_df[row.name].mode()[0])

for incomplete_categorical_feature in incomplete_categorical_features:

    all_null = all_null.drop(incomplete_categorical_feature, inplace=True)



# Do a linear regression with all nominal features

just_nominal_features = []

for number, feature, in enumerate(list(house_test_df.columns.values)):

    if number > 0:

        if is_numeric_dtype(house_test_df[feature]):

            just_nominal_features.append(feature)

test_linear_x = house_test_df[just_nominal_features]

test_linear_predictions = everything_linear_model.predict(test_linear_x)

test_linear_predictions

my_submission = pd.DataFrame({'Id': house_test_df.Id, 'SalePrice': test_linear_predictions})

my_submission.to_csv('submission.csv', index=False)

print("done!")