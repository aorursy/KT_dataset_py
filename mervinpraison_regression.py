# Import numpy and pandas

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Read the CSV file into a DataFrame: df

df = pd.read_csv('../input/gapminder.csv')



# Create arrays for features and target variable

y = df.life

X = df.fertility



# Print the dimensions of X and y before reshaping

print("Dimensions of y before reshaping: {}".format(y.values.shape))

print("Dimensions of X before reshaping: {}".format(X.values.shape))



# Reshape X and y

y = y.values.reshape(-1, 1)

X = X.values.reshape(-1, 1)



# Print the dimensions of X and y after reshaping

print("Dimensions of y after reshaping: {}".format(y.shape))

print("Dimensions of X after reshaping: {}".format(X.shape))
df.head()
import seaborn as sns

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), square=True, cmap='RdYlGn', fmt= '.1f', ax=ax);
# Import LinearRegression

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



# Create the regressor: reg

reg = LinearRegression()



X_fertility = X.copy()



# Create the prediction space

prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)



# Fit the model to the data

reg.fit(X_fertility, y)



# Compute predictions over the prediction space: y_pred

y_pred = reg.predict(prediction_space)



# Print R^2 

print(reg.score(X_fertility, y))



# Plot regression line

plt.scatter(X_fertility, y, c=y, alpha=.7)

plt.plot(prediction_space, y_pred, color='black', linewidth=3)

plt.tight_layout()

plt.show();
# Import necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)



# Create the regressor: reg_all

reg_all = LinearRegression()



# Fit the regressor to the training data

reg_all.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))



rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
X_train.shape
y_train.shape
X.shape
# Import the necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# Create a linear regression object: reg

reg = LinearRegression()



# Compute 5-fold cross-validation scores: cv_scores

cv_scores = cross_val_score(reg, X, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores)



print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Import necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



# Create a linear regression object: reg

reg = LinearRegression()



# Perform 3-fold CV

cvscores_3 = cross_val_score(reg , X, y, cv=3)

print(np.mean(cvscores_3))



# Perform 10-fold CV

cvscores_10 = cross_val_score(reg , X, y, cv=10)

print(np.mean(cvscores_10))
y = df.life.values
y.shape
X = df.drop(['life', 'Region'], axis=1)
X.shape
df_columns = X.columns

X = X.values
# Import Lasso

from sklearn.linear_model import Lasso



# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=.4, normalize=True)



# Fit the regressor to the data

lasso.fit(X,y)



# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)



# Plot the coefficients

plt.plot(range(len(df_columns)), lasso_coef)

plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)

plt.margins(0.02)

plt.show()
df_columns
def display_plot(cv_scores, cv_scores_std):

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    ax.plot(alpha_space, cv_scores)



    std_error = cv_scores_std / np.sqrt(10)



    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

    ax.set_ylabel('CV Score +/- Std Error')

    ax.set_xlabel('Alpha')

    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

    ax.set_xlim([alpha_space[0], alpha_space[-1]])

    ax.set_xscale('log')

    plt.show()
# Import necessary modules

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score



# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 50)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge,X,y, cv=10)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the plot

display_plot(ridge_scores, ridge_scores_std)
