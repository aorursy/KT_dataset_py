import pandas as pd
# save filepath to variable for easier access
file_path = '../input/train.csv'
# read the data and store data in DataFrame titled data
data = pd.read_csv(file_path) 
# print a data summary here
data.describe()

# The mean sale price is:
data.describe()['SalePrice']['mean'].round()
# The year the oldest house was built is:
data.describe()['YearBuilt']['min']
import matplotlib.pyplot as plt
plt.hist(data.SalePrice, bins=50);
# plot a histogram of the YearBuilt column here
plt.hist(data.YearBuilt, bins=50);
y = data.SalePrice
# create a list of numeric variables called predictors
predictors = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YrSold']
# create a DataFrame called X containing the predictors here
X = data[predictors]
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
# fit your decision tree model here
model.fit(X,y);
# make predictions with your model here
predictions = model.predict(X)
# compare the model's predictions with the true sale prices of the first few houses here
X.assign(Prediction = predictions).assign(Y = y).head()
from sklearn.metrics import mean_absolute_error
# compute the mean absolute error of your predictions here
mean_absolute_error(y, predictions)
# compute the mean absolute error on the validation data here
from sklearn.model_selection import train_test_split
val_model = DecisionTreeRegressor()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
val_model.fit(train_X, train_y)
val_predictions = val_model.predict(val_X)
mean_absolute_error(val_y, val_predictions)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
reg.coef_
array([ 0.5,  0.5])
# make predictions for the test data here
test = pd.read_csv('../input/test.csv')
test_X = test[predictors]
test_predictions = model.predict(test_X)
# prepare your submission file here
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})
submission.to_csv('submission.csv', index=False)