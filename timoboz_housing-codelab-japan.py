import pandas as pd
# save filepath to variable for easier access
file_path = '../input/train.csv'
# read the data and store data in DataFrame titled data
data = pd.read_csv(file_path) 
# print a data summary here

# The mean sale price is:

# The year the oldest house was built is:

import matplotlib.pyplot as plt
plt.hist(data.SalePrice, bins=50);
# plot a histogram of the YearBuilt column here

y = data.SalePrice
# create a list of numeric variables called predictors

# create a DataFrame called X containing the predictors here

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
# fit your decision tree model here

# make predictions with your model here

# compare the model's predictions with the true sale prices of the first few houses here

from sklearn.metrics import mean_absolute_error
# compute the mean absolute error of your predictions here

# compute the mean absolute error on the validation data here

# make predictions for the test data here

# prepare your submission file here
