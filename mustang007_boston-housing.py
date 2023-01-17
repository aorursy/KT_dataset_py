
from sklearn.datasets import load_boston

boston_dataset = load_boston()

import numpy as np 
import pandas as pd 

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

%matplotlib inline


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
#The target values is missing from the data. Create a new column of target values and add it to dataframe


boston['MEDV'] = boston_dataset.target
boston.head()
boston.info()
boston.describe()
boston.nunique()
# to check null values

boston.isnull().sum()
boston['MEDV'].plot()
# to check skewnes
from scipy.stats import skew
boston['MEDV'].skew()
100
np.log1p(100)
(np.log1p(boston['MEDV'])).skew()
# compute the pair wise correlation for all columns  
correlation_matrix = boston.corr().round(2)
plt.figure(figsize=(10,6))
sns.heatmap(data=correlation_matrix, annot=True)

a = [1,2,3,]

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
X = boston.drop(columns=['RAD','MEDV'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from  sklearn.linear_model import Lasso
L_model = Lasso(alpha=0.1)
L_model.fit(X_train, Y_train)
# model evaluation for testing set

L = L_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso_reg = Lasso()
parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],
              "fit_intercept": [True, False],
             }
grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(X_train, Y_train)
grid.best_params_
grid.best_score_
L_model = Lasso(alpha=0.3)
L_model.fit(X_train, Y_train)
# model evaluation for testing set

y_test_predict = L_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import Ridge
R_model = Ridge()
R_model.fit(X_train, Y_train)
# model evaluation for testing set

r = R_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
lasso_reg = Ridge()
parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],
              "fit_intercept": [True, False],
             }
grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(X_train, Y_train)
grid.best_params_
R_model = Ridge(alpha=10)
R_model.fit(X_train, Y_train)
# model evaluation for testing set

y_test_predict = R_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.linear_model import ElasticNet
E_model = Ridge()
E_model.fit(X_train, Y_train)
# model evaluation for testing set

e = E_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
final = L*0.1 + r*0.6 + e*0.3
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, final)))

# r-squared score of the model
r2 = r2_score(Y_test, final)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# Difference between L1 and L2 regularization
# L1 Regularization
# L1 penalizes sum of absolute value of weights.
# L1 has a sparse solution
# L1 has multiple solutions
# L1 has built in feature selection
# L1 is robust to outliers
# L1 generates model that are simple and interpretable but cannot learn complex patterns


# L2 Regularization
# L2 regularization penalizes sum of square weights.
# L2 has a non sparse solution
# L2 has one solution
# L2 has no feature selection
# L2 is not robust to outliers
# L2 gives better prediction when output variable is a function of all input features
# L2 regularization is able to learn complex data patterns