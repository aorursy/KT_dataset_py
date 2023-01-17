import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # For distribution plotting
import os

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

print('train has {} rows and {} features'.format(train.shape[0],train.shape[1]))
print('test has {} rows and {} features'.format(test.shape[0],test.shape[1]))


train_cols_withnull = len(train.columns[train.isna().any()])
test_cols_withnull = len(test.columns[test.isna().any()])

print('train has {} columns with null values'.format(train_cols_withnull))
print('test has {} columns with null values'.format(test_cols_withnull))

import seaborn as sns

# Distribution plot
sns.distplot(train['SalePrice'], kde=False)
plt.title("Distribution of SalePrice", fontsize=20, y=1.012)

# Probability plot
plt.figure()
stats.probplot(train['SalePrice'], rvalue=True, plot=plt)
plt.title("Probability plot", fontsize=20, y=1.012)
plt.ylabel("SalePrice")


# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

# applying log transformation
sale_price_log = np.log(train['SalePrice'])

# Distribution plot
sns.distplot(sale_price_log, kde=False)

# Probability plot
plt.figure()
stats.probplot(sale_price_log, rvalue=True, plot=plt)
plt.title("Probability plot", fontsize=20, y=1.012)
plt.ylabel("SalePrice")

print("Skewness: %f" % sale_price_log.skew())
print("Kurtosis: %f" % sale_price_log.kurt())


# Combine train and test data sets
train_and_test = pd.concat([train.iloc[:,:-1],test],axis=0)

# drop ID column
train_and_test = train_and_test.drop(columns=['Id'],axis=1)
# Fill null categorical data with ‘Unknown’
# Fill numeric columns with median
for column in train_and_test.columns:
    dtype = train_and_test[column].dtype
    if dtype in ['int64', 'float64']:
        median = train_and_test[column].median()
        train_and_test[column].fillna(value = median, inplace=True)
    if dtype == 'object':
        train_and_test[column].fillna(value = 'UNKNOWN', inplace=True)
        
train_and_test = pd.get_dummies(train_and_test)
# Note: train_data will not have SalePrice, but SalePrice can be referenced from the original data set
train_data = train_and_test.iloc[:1460,:]
test_data = train_and_test.iloc[1460:,:]

print('train df has {} rows and {} features'.format(train_data.shape[0],train_data.shape[1]))
print('test df has {} rows and {} features'.format(test_data.shape[0],test_data.shape[1]))
X = train_data
y = np.log(train.SalePrice)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression().fit(X_train, y_train)
predictions = linear_regression.predict(X_test)

# Evaluate efficiency
features = np.sum(linear_regression.coef_ != 0)
print("number of features used:   ", features)
print("mean squared error:        ", mean_squared_error(y_test, predictions))
print("training score:            ", linear_regression.score(X_train,y_train) )
print("test score:                ", linear_regression.score(X_test,y_test))

# MSE of test > MSE of train => OVER FITTING of the data.
# MSE of test < MSE of train => UNDER FITTING of the data
plt.scatter(predictions, y_test, alpha=.75, color='b')

plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.show()
final_predictions = np.exp(linear_regression.predict(test_data))
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': final_predictions})
submission.to_csv('linear_regression.csv', index=False)
# iterate lambdas
from sklearn.linear_model import Ridge

best_ridge_model = None
best_mse = 1

# Iterate through alpha values 0.01 to 0.2 by every 0.001
for alpha in np.arange(0.1, 0.2, 0.001):
    ridge_regression = Ridge(alpha=alpha)
    ridge_regression.fit(X_train, y_train)
    prediction = ridge_regression.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    if (mse < best_mse):
        best_mse = mse
        best_ridge_model = ridge_regression
    

features = np.sum(best_ridge_model.coef_ != 0)
print("best model found at alpha =", best_ridge_model.alpha)
print("number of features used:   ", features)
print("mean squared error:        ", best_mse)
print("training score:            ", best_ridge_model.score(X_train,y_train) )
print("test score:                ", best_ridge_model.score(X_test,y_test))

# MSE of test > MSE of train => OVER FITTING of the data.
# MSE of test < MSE of train => UNDER FITTING of the data
final_predictions = np.exp(best_ridge_model.predict(test_data))
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': final_predictions})
submission.to_csv('ridge_regression.csv', index=False)
submission.head()

from sklearn.linear_model import Lasso

best_lasso_model = None
best_mse = 1

# Iterate through alpha values 0.0001 to 0.01 by every 0.0001
for alpha in np.arange(0.0001, 0.01, 0.0001):
    lasso_regression = Lasso(alpha=alpha, max_iter=10e5).fit(X_train,y_train)
    prediction = lasso_regression.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    if (mse < best_mse):
        best_mse = mse
        best_lasso_model = lasso_regression

features = np.sum(best_lasso_model.coef_ != 0)
print("best model found at alpha =", best_lasso_model.alpha)
print("number of features used:   ", features)
print("mean squared error:        ", best_mse)
print("training score:            ", best_lasso_model.score(X_train,y_train))
print("test score:                ", best_lasso_model.score(X_test,y_test))

# MSE of test > MSE of train => OVER FITTING of the data.
# MSE of test < MSE of train => UNDER FITTING of the data

final_predictions = np.exp(best_lasso_model.predict(test_data))
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': final_predictions})
submission.to_csv('lasso_regression.csv', index=False)