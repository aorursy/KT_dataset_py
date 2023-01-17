# Import the necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
# Load the CSV file containing the experience vs salary data into a pandas dataframe

df = pd.read_csv("../input/sample-salary-data-for-simple-linear-regression/Salary_Data.csv")



# Inspect the first 10 rows

df.head(10)
# Just a small adjustment in the column names

df.rename(columns = {"YearsExperience" : "years", "Salary" : "salary"}, inplace = True)



# Inspect the data again

df.head()
# Review some of the important statistics of the data

df.describe()
# Check the correlation coefficient between years and salary

df.corr()
# Let's display a regression plot to visualise our data

plt.figure(figsize = (12, 6))

sns.regplot(data = df, x = "years", y = "salary", color = "red")



# Label the plot

plt.title("Regression Plot: Years of Experience vs Salary")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
# Set years as the feature set (X) and salary as the target set (y)

X, y = np.array(df[["years"]]), np.array(df[["salary"]])

# X is being assigned a 2D array (by using double square brackets) because the fit method I'm going to use later to train

# the model expects the feature matrix to be 2D.

# The same is optional for the target matrix in case it is a single variable, but I'm doing this for making sure that

# I don't forget it when I'll be dealing with multiple targets



# Check the shapes of the above

print(X.shape, y.shape)
# I'll be using train-test split on the above data for obtaining the training data and the testing data

trainX, testX, trainy, testy = train_test_split(X, y, test_size = 1/4)



# Review the splitted data

print(trainX, trainy)

print(testX, testy)
# Instantiate the linear regression object

regr = LinearRegression()
# Fit (train) the model with our training data

regr.fit(trainX, trainy)



# Print the coefficients matrix and the intercepts vector of the regression line

print(regr.coef_, regr.intercept_)
# Predict the values for testX

# I'll assign the array of predicted values to a new variable testyCap

testyCap = regr.predict(testX)



# View the array

testyCap
# Let's compare the true values versus the predicted ones for the test set

# For this purpose, let's create a dataframe

dfComparison = pd.DataFrame(dict(X = testX[:, 0], trueValues = testy[:, 0], predicted = testyCap[:, 0]))

dfComparison
# Now I'll evaluate the model using various statistical methods



# Mean Absolute Error

mae = np.mean(np.absolute(testyCap - testy))

# Mean Squared Error

mse = np.mean((testyCap - testy) ** 2)

# Root Mean Squared Error

rmse = np.sqrt(mse)

# Relative Absolute Error

rae = np.sum(np.absolute(testyCap - testy)) / np.sum(np.absolute(testy - np.mean(testy)))

# Relative Squared Error

rse = np.sum((testyCap - testy) ** 2) / np.sum((testy - np.mean(testy)) ** 2)

# R2 Score (calculated)

r2 = 1 - rse

# R2 Score (using method)\

r2M = r2_score(testy, testyCap)

# Variance score (equal to r2)

varScore = regr.score(testX, testy)
# Print the stats

print("""/

MAE: %f

MSE: %f

RMSE: %f

RAE: %f

RSE: %f

R2 Score (calculated): %f

R2 Score (using method): %f

Variance Score: %f

"""%(mae, mse, rmse, rae, rse, r2, r2M, varScore))
# Let's visualise the trainX, trainy and testX, testy separately

fig, axes = plt.subplots(2, 1, figsize = (10, 10))

ax1, ax2 = axes

fig.suptitle("Regression Plots for Two Different Sets")



# For training set

sns.regplot(trainX, trainy, color = "blue", ax = ax1)

ax1.set_title("Training Set")

ax1.set_xlabel("trainX")

ax1.set_ylabel("trainy")



# For test set

sns.regplot(testX, testy, color = "red", ax = ax2)

ax2.set_title("Test Set")

ax2.set_xlabel("testX")

ax2.set_ylabel("testy")



plt.show()
# Let's predict our model for some sample data

sampleX = np.array([1, 2, 1.7, 3, 2.5, 0.2, 0, 9.1, 5.6, 6.3, 4.9])

sampley = regr.coef_[0, 0] * sampleX + regr.intercept_[0]



# Feed it into a dataframe

dfSample = pd.DataFrame({"X" : sampleX, "y" : sampley})

# View it

dfSample
# Visualising just for fun

dfSample.plot(kind = "scatter", x = "X", y = "y", color = "blue", figsize = (10, 8), marker = "+")

plt.plot(sampleX, sampley, color = "red", alpha = 0.4)

plt.show()