 ##THE LYBRARIES USED IN THIS NOTEBOOK

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#To avoid FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

import seaborn as sns

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import r2_score



                               
dataset_link = '../input/kc_house_data.csv'

houses_df = pd.read_csv(dataset_link)



minimum_y = houses_df['price'].min()

# s refers to size

#alpha --> 0.0 transparent through 1.0 opaque

plt.figure(figsize=(10,10))

plt.scatter(x = houses_df.sqft_living,y = houses_df.price/minimum_y,s = 1, alpha = 1)

plt.xlabel("SQFT_LIVING")

plt.ylabel("PRICES")

plt.title("SEATTLE HOME PRICES")

plt.yscale('log')



#Lets see what we have in the dataset

houses_df.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x) #supress scientific notation

houses_df.describe().iloc[:,1:].drop(['yr_built','yr_renovated','zipcode'],axis=1)



correlation = houses_df.iloc[:,2:].corr(method='pearson')

correlation.style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)

correlation.price.sort_values(ascending=False)[1:]

#we drop the first value as it is with itself.
correlated_variables = correlation.idxmin()

correlation_values = correlation.min().values



correlation_dict = {'First Variable':correlated_variables.index, 'Second Variable':correlated_variables.values, 'Values':correlation_values}

pd.DataFrame(correlation_dict)

sns.set(style = "ticks", color_codes=True)

correlation_features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','yr_built']

sns.set_style("darkgrid")

sns.pairplot(houses_df[correlation_features],diag_kind="kde",dropna=True)

#diag_kind:Use kernel density estimates for univariate plots:

#kind:Fit linear regression models to the scatter plots

plt.figure(figsize=(25,20))

plt.show();
dataset_train, dataset_test, price_train, price_test = train_test_split(houses_df,houses_df['price'],test_size=0.2,random_state=3)

#Build the regression model using only sqft_living as a feature

# Create linear regression object

regression_ols = linear_model.LinearRegression()

#We convert the column sqft_living to a numpy array to make it easier to work

living_train = np.asarray(dataset_train.sqft_living)

living_train = living_train.reshape(-1,1)

#Train the model using the training sets

#Here price is the "target" data in this model, the other features are the independet variables

ols_model = regression_ols.fit(living_train, price_train)

living_test = np.asarray(dataset_test.sqft_living)

living_test = living_test.reshape(-1,1)

#We the trained dataset we make a prediction for the test dataset

prediction_test_ols = ols_model.predict(living_test)



print ('Ordinary Least Squares')

#Coefficient

print('Coefficient:',ols_model.coef_[0])

print ('Intercept', ols_model.intercept_)

# Apply the model we created using the training data to the test data, and calculate the RSS.

print('RSS',((price_test - prediction_test_ols) **2).sum())

# Calculate the RMSE ( Root Mean Squared Error)

print('RMSE', np.sqrt(metrics.mean_squared_error(price_test,prediction_test_ols)))

#The model's performance on test set is:

print('The model\'s performance is %.2f\n'% ols_model.score(living_test, price_test))







living_test_sort = np.sort(living_test.reshape(-1))

plt.scatter(living_test, price_test, color='blue', alpha=0.25,label='Real Price')

#When you plot you have to sort the array, in this case square feet living , the one that belongs to the test, if you dont do this, the plot looks weird

plt.plot(living_test_sort, ols_model.predict(living_test_sort.reshape(-1,1)),'r--',linewidth=3, label='Ordinary Least squares Regression')





plt.xlabel('Price')

plt.ylabel('Square_feet_living')

plt.legend()

plt.yscale('log')

plt.figure(figsize=(15,10))











#Blue dots are from the original data the red line is the prediction from the least squares 





plt.show()



actual_predicted_data_ols = pd.DataFrame({'Actual': price_test, 'Predicted': np.round(prediction_test_ols,decimals=3)})

actual_predicted_data_ols.head()

regression_lasso = linear_model.Lasso(alpha=.1)

lasso_model = regression_lasso.fit(living_train, price_train)

prediction_test_lasso = lasso_model.predict(living_test)



print ('Lasso Regression')

#Intercept

print ('Intercept', lasso_model.intercept_)

# Coefficient

print('Coefficient:', lasso_model.coef_[0])

# Apply the model we created using the training data to the test data, and calculate the RSS.

print('RSS',((price_test - prediction_test_lasso) **2).sum())

# Calculate the RMSE (Root Mean Squared Error)

print('RMSE', np.sqrt(metrics.mean_squared_error(price_test,prediction_test_lasso)))

# Coefficient of determination R^2 of the prediction

print('The model\'s performance is %.2f\n' % lasso_model.score(living_test, price_test))

# Plot 

plt.scatter(living_test, price_test, color='green', alpha=0.25,label='Real Price')

plt.plot(living_test_sort, lasso_model.predict(living_test_sort.reshape(-1,1)),'b--',linewidth=3, label='Lasso Regression')

plt.xlabel('Price')

plt.ylabel('square_feet_living')

plt.legend()

plt.yscale('log')

plt.figure(figsize=(15,10))



plt.show()



actual_predicted_data_lasso = pd.DataFrame({'Actual': price_test, 'Predicted': np.round(prediction_test_lasso,decimals=3)})



actual_predicted_data_lasso.head()
regression_ridge = linear_model.Ridge(alpha=[.1])

ridge_model = regression_ridge.fit(living_train, price_train)

prediction_test_ridge = ridge_model.predict(living_test)



print ('Ridge Regression')

#Intercept

print ('Intercept', ridge_model.intercept_)

# Coeficient

print('Coefficient:', ridge_model.coef_[0])

# Apply the model we created using the training data to the test data, and calculate the RSS.

print('RSS',((price_test - prediction_test_ridge) **2).sum())

# Calculate the RMSE (Root Mean Squared Error)

print('RMSE', np.sqrt(metrics.mean_squared_error(price_test,prediction_test_ridge)))

# Coefficient of determination R^2 of the prediction

print('The model\'s performance is %.2f\n' % ridge_model.score(living_test, price_test))

# Plot 

plt.scatter(living_test, price_test, color='brown', alpha=0.25,label='Real Price')

plt.plot(living_test_sort, ridge_model.predict(living_test_sort.reshape(-1,1)),'g--',linewidth=3, label='Ridge Regression')

plt.xlabel('Price')

plt.ylabel('square_feet_living')

plt.legend()

plt.yscale('log')

plt.figure(figsize=(15,10))



plt.show()
actual_predicted_data_ridge = pd.DataFrame({'Actual': price_test, 'Predicted': np.round(prediction_test_ridge,decimals=3)})

actual_predicted_data_ridge.head()
#n_estimators: It controls the number of weak learners.

#learning_rate:Controls the contribution of weak learners in the final combination. There is a trade-off between learning_rate and n_estimators.

#base_estimators: It helps to specify different ML algorithm. By default sklearn uses decision tree

adaboost_regressor = AdaBoostRegressor(n_estimators=1500, learning_rate = 0.001, loss='exponential')

ada_model = adaboost_regressor.fit(living_train, price_train)

prediction_test_ada = ada_model.predict(living_test)

# Apply the model we created using the training data to the test data, and calculate the RSS.

print('RSS',((price_test - prediction_test_ada) **2).sum())

# Calculate the RMSE (Root Mean Squared Error)

print('RMSE', np.sqrt(metrics.mean_squared_error(price_test,prediction_test_ada)))

#Coefficient of determination R^2 of the prediction

print('The model\'s performance is %.2f\n' % ada_model.score(living_test, price_test))

# Plot 

plt.scatter(living_test, price_test, color='black', alpha=0.25,label='Real Price')

plt.plot(living_test_sort, ada_model.predict(living_test_sort.reshape(-1,1)),'g--',linewidth=3, label='AdaBoost regressor')

plt.xlabel('Price')

plt.ylabel('square_feet_living')

plt.legend()

plt.yscale('log')

plt.figure(figsize=(15,10))



plt.show()



actual_predicted_data_ada = pd.DataFrame({'Actual': price_test, 'Predicted': np.round(prediction_test_ada,decimals=3)})

actual_predicted_data_ada.head()

dataset_train, dataset_test = train_test_split(houses_df,test_size=0.2,random_state=3)

price_train= np.asarray(dataset_train.price).reshape(-1,1)

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','yr_built','zipcode']

train_matrix = dataset_train.as_matrix(my_features)

regr_with_more_features = linear_model.LinearRegression()

model_least_squares = regr_with_more_features.fit(train_matrix, price_train)





matrix_test = dataset_test.as_matrix(my_features)



price_test_multiple_regression = np.asarray(dataset_test.price).reshape(-1,1)



prediction_test_least_squares = model_least_squares.predict(matrix_test)



print ('Least Squares Means')

#Coefficient

print('Coefficient:',model_least_squares.coef_[0])

#Apply the model we created using the training data to the test data, and calculate the RSE.

print('RSS',((price_test_multiple_regression - prediction_test_least_squares) **2).sum())

# Calculate the MSE

print('RMSE', np.sqrt(metrics.mean_squared_error(price_test_multiple_regression,prediction_test_least_squares)))

# Coefficient of determination R^2 of the prediction

print('The model\'s performance is %.2f\n' % model_least_squares.score(matrix_test, price_test_multiple_regression))



price_test_multiple_regression = price_test_multiple_regression.reshape(-1)

prediction_test_least_squares  = prediction_test_least_squares.reshape(-1) 



actual_predicted_data_least_squares = pd.DataFrame({'Actual': price_test_multiple_regression, 'Predicted': np.round(prediction_test_least_squares,decimals=3)})

actual_predicted_data_least_squares.head()