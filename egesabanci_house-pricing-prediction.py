import pandas as pd

import numpy as np
# data paths

TRAIN_PATH = "../input/house-prices-advanced-regression-techniques/train.csv"

TEST_PATH = "../input/house-prices-advanced-regression-techniques/test.csv"



# train and test data

data_train = pd.DataFrame(pd.read_csv(TRAIN_PATH)) # train set

data_test = pd.DataFrame(pd.read_csv(TEST_PATH))   # test set



# preview of train set

data_train.head(10)
# drop nan values from columns - threshold = 50

data_train = data_train.dropna(axis = 1, thresh = 1410)  # train data cols

data_test  = data_test.dropna(axis = 1, thresh = 1410)   # test data cols



# drop nan values from columns - how = "any"

data_train = data_train.dropna(axis = 0, how = "any")    # train data rows

data_test  = data_test.dropna(axis = 0, how = "any")      # test data rows
def convert_numerical(dataframe):

    

    # column names in dataframe

    col_name = [i for i in dataframe]

    

    for i in col_name:

        for a in dataframe[i]:

            if type(a) == str:

                dataframe[i], _ = pd.factorize(dataframe[i])

                break

    

    return dataframe



# convert all string data into numerical and categorical data

data_train = convert_numerical(data_train)

data_test  = convert_numerical(data_test)
data_train.head(10)
# we will store our models in this dict

models = dict()
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
x = data_train.drop(labels = ["SalePrice"], axis = 1)   # indepent variable

y = data_train["SalePrice"]                             # depent variable (prediction vals)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, train_size = 0.7)
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(x_train, y_train)
# linear regression model coef and model intercept

print("Model Coefficients:", lr_model.coef_, "\nModel Intercept:", lr_model.intercept_)
# linear regression model loss and model score

lr_model_score = r2_score(y_test, lr_model.predict(x_test))

lr_model_loss  = mean_squared_error(y_test, lr_model.predict(x_test))



# model score and loss

print("Model score:", lr_model_score, "\nModel Loss:", lr_model_loss)



# add loss and error to models dictionary

models["LinearRegression"] = {"model_score": lr_model_score, "model_loss": lr_model_loss}
from sklearn.neural_network import MLPRegressor

ann_model = MLPRegressor().fit(x_train, y_train)
# ann model loss and score

ann_model_score = r2_score(y_test, ann_model.predict(x_test))

ann_model_loss = mean_squared_error(y_test, ann_model.predict(x_test))



# loss and score for ann model

print("Model Loss:", ann_model_loss, "\nModel Score:", ann_model_score)
# add to the models dictionary

models["ANN Model"] = {"model_loss": ann_model_loss, "model_score": ann_model_score}
import matplotlib.pyplot as plt



plt.figure(figsize = (10, 5))

plt.grid(True)

plt.title("Accuracy", fontsize = 30)

plt.bar(["Linear Regression"], models["LinearRegression"]["model_score"])

plt.bar(["ANN Regression"], models["ANN Model"]["model_score"])

plt.legend(["Linear Regression", "ANN Regression"])
plt.figure(figsize = (10, 5))

plt.grid(True)

plt.title("Loss", fontsize = 30)

plt.bar(["Linear Regression"], models["LinearRegression"]["model_loss"])

plt.bar(["ANN Regression"], models["ANN Model"]["model_loss"])

plt.legend(["Linear Regression", "ANN Regression"])
# all test set predictions

test_set_predictions = lr_model.predict(data_test)
# plot test predictions



plt.figure(figsize = (10, 5))

plt.grid(True)

plt.title("All Test Predictions", fontsize = 30)

plt.xlabel("Number of Data for Prediction = 1397", fontsize = 20)

plt.ylabel("Predictions = 1397 Houses", fontsize = 20)

plt.scatter([x for x in range(len(test_set_predictions))], test_set_predictions)

plt.show()