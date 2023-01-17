# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
# Import numpy and pandas

import numpy as np

import pandas as pd



# Read the CSV file into a DataFrame: df

df = pd.read_csv('../input/gapminder.csv')

df.columns









# Create arrays for features and target variable

y = df.life.values

X_fertility= df.fertility.values

X= df[['population','fertility','GDP']].values



y=y.reshape(-1,1)

X_fertility=X_fertility.reshape(-1,1)

X= X.reshape(-1,3)

print(y.shape,X_fertility.shape,X.shape)
# Import LinearRegression

from sklearn.linear_model import LinearRegression



# Create the regressor: reg

reg = LinearRegression()



# Create the prediction space

prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)





# Fit the model to the data

reg.fit(X_fertility,y)



# Compute predictions over the prediction space: y_pred

y_pred = reg.predict(prediction_space)



# Print R^2 

print(reg.score(X_fertility, y))



# Plot regression line

plt.scatter(x=df.fertility.values,y=df.life.values,color='blue')

plt.plot(prediction_space, y_pred, color='black', linewidth=3)

plt.show()
# Import necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.3, random_state=42)



# Create the regressor: reg_all

reg_all = LinearRegression()



# Fit the regressor to the training data

reg_all.fit(X_train,y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(reg_all, X, y, cv=5)

print(cv_scores)

print(np.mean(cv_scores))
# Perform 3-fold CV

cvscores_3 = cross_val_score(reg,X,y,cv=3)

print(np.mean(cvscores_3))



# Perform 10-fold CV

cvscores_10 = cross_val_score(reg,X,y,cv=10)

print(np.mean(cvscores_10))

%timeit cvscores_3
%timeit cvscores_10
# Import Lasso

from sklearn.linear_model import Lasso



# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=0.4,normalize=True)

X=df.drop(['life','Region'],axis=1).values

y=df.life.values

X=X.reshape(-1,8)





# Fit the regressor to the data

lasso.fit(X,y)



# Compute and print the coefficients

lasso_coef = lasso.fit(X,y).coef_

print(lasso_coef)

df_columns= df.drop(['life','Region'],axis=1).columns

plt.plot(range(len(df_columns)), lasso_coef)

plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)

plt.margins(0.02)

plt.show()
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

    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the plot

display_plot(ridge_scores, ridge_scores_std)

# Import necessary modules

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, train_test_split



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Create the hyperparameter grid

l1_space = np.linspace(0, 1, 30)

param_grid = {'l1_ratio': l1_space}



# Instantiate the ElasticNet regressor: elastic_net

elastic_net = ElasticNet()



# Setup the GridSearchCV object: gm_cv

gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)



# Fit it to the training data

gm_cv.fit(X_train,y_train)



# Predict on the test set and compute metrics

y_pred = gm_cv.predict(X_test)

r2 = gm_cv.score(X_test, y_test)

mse = mean_squared_error(y_test, y_pred)

print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))

print("Tuned ElasticNet R squared: {}".format(r2))

print("Tuned ElasticNet MSE: {}".format(mse))

df.boxplot( 'life','Region', rot=60)



# Show the plot

plt.show()
df_region = pd.get_dummies(df,drop_first=True)



# Print the new columns of df_region

print(df_region.columns)
from sklearn.linear_model import Ridge



# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=0.5,normalize=True)



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge,X,y,cv=5)



# Print the cross-validated scores

print(ridge_cv)