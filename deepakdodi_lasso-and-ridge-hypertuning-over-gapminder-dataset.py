import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
np.set_printoptions(suppress=True)
data = pd.read_csv('../input/gapminder.csv')
data.dtypes
data.head(5)
data = data.drop(['Region'], axis =1)
data.isnull().sum()
X = data.drop(['life'], axis = 1)
y = data.life
X.head()
y.head()
Training_Accuracy_Before = []
Testing_Accuracy_Before = []
Training_Accuracy_After = []
Testing_Accuracy_After = []
Models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
logreg = LinearRegression()
logreg.fit(X_train, y_train)

train_score = logreg.score(X_train, y_train)
print(train_score)
test_score = logreg.score(X_test, y_test)
print(test_score)

Training_Accuracy_Before.append(train_score)
Testing_Accuracy_Before.append(test_score)
alpha_space = np.logspace(-4, 0, 30)   # Checking for alpha from .0001 to 1 and finding the best value for alpha
alpha_space
ridge_scores = []
ridge = Ridge(normalize = True)
for alpha in alpha_space:
    ridge.alpha = alpha
    val = np.mean(cross_val_score(ridge, X, y, cv = 10))
    ridge_scores.append(val)
lasso_scores = []
lasso = Lasso(normalize = True)
for alpha in alpha_space:
    lasso.alpha = alpha
    val = np.mean(cross_val_score(lasso, X, y, cv = 10))
    lasso_scores.append(val)
plt.figure(figsize=(8, 8))
plt.plot(alpha_space, ridge_scores, marker = 'D', label = "Ridge")
plt.plot(alpha_space, lasso_scores, marker = 'D', label = "Lasso")
plt.legend()
plt.show()
# Performing GridSearchCV with Cross Validation technique on Lasso Regression and finding the optimum value of alpha

params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
lasso = Lasso(normalize=True)
lasso_model = GridSearchCV(lasso, params, cv = 10)
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)
# Using value of alpha as 0.0000171 to get best accuracy for Lasso Regression
lasso = Lasso(alpha = 0.0000171, normalize = True)
lasso.fit(X_train, y_train)

train_score = lasso.score(X_train, y_train)
print(train_score)
test_score = lasso.score(X_test, y_test)
print(test_score)

Training_Accuracy_Before.append(train_score)
Testing_Accuracy_Before.append(test_score)
# Performing GridSearchCV with Cross Validation technique on Ridge Regression and finding the optimum value of alpha

params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
ridge = Ridge(normalize=True)
ridge_model = GridSearchCV(ridge, params, cv = 10)
ridge_model.fit(X_train, y_train)
print(ridge_model.best_params_)
print(ridge_model.best_score_)
# Using value of alpha as 0.020092 to get best accuracy for Ridge Regression
ridge = Ridge(alpha = 0.020092, normalize = True)
ridge.fit(X_train, y_train)

train_score = ridge.score(X_train, y_train)
print(train_score)
test_score = ridge.score(X_test, y_test)
print(test_score)

Training_Accuracy_Before.append(train_score)
Testing_Accuracy_Before.append(test_score)
coefficients = lasso.coef_
coefficients
plt.figure(figsize = (10, 6))
plt.plot(range(len(X_train.columns)), coefficients)
plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation = 90)
plt.show()
X_train.columns
X = data[['fertility', 'HIV', 'CO2', 'BMI_male', 'GDP', 'BMI_female', 'child_mortality']]
y = data.life
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
logreg = LinearRegression()
logreg.fit(X_train, y_train)

train_score = logreg.score(X_train, y_train)
print(train_score)
test_score = logreg.score(X_test, y_test)
print(test_score)

Training_Accuracy_After.append(train_score)
Testing_Accuracy_After.append(test_score)
# Performing GridSearchCV with Cross Validation technique on Lasso Regression and finding the optimum value of alpha

params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
lasso = Lasso(normalize=True)
lasso_model = GridSearchCV(lasso, params, cv = 10)
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)
# Using value of alpha as 0.000705 to get best accuracy for Lasso Regression
lasso = Lasso(alpha = 0.000705, normalize = True)
lasso.fit(X_train, y_train)

train_score = lasso.score(X_train, y_train)
print(train_score)
test_score = lasso.score(X_test, y_test)
print(test_score)

Training_Accuracy_After.append(train_score)
Testing_Accuracy_After.append(test_score)
# Performing GridSearchCV with Cross Validation technique on Ridge Regression and finding the optimum value of alpha

params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
ridge = Ridge(normalize=True)
ridge_model = GridSearchCV(ridge, params, cv = 10)
ridge_model.fit(X_train, y_train)
print(ridge_model.best_params_)
print(ridge_model.best_score_)
# Using value of alpha as 0.020092 to get best accuracy for Ridge Regression
ridge = Ridge(alpha = 0.020092, normalize = True)
ridge.fit(X_train, y_train)

train_score = ridge.score(X_train, y_train)
print(train_score)
test_score = ridge.score(X_test, y_test)
print(test_score)

Training_Accuracy_After.append(train_score)
Testing_Accuracy_After.append(test_score)
plt.plot(Training_Accuracy_Before, label = 'Training_Accuracy_Before')
plt.plot(Training_Accuracy_After, label = 'Training_Accuracy_After')
plt.xticks(range(len(Models)), Models, Rotation = 45)
plt.title('Training Accuracy Behaviour')
plt.legend()
plt.show()
plt.plot(Testing_Accuracy_Before, label = 'Testing_Accuracy_Before')
plt.plot(Testing_Accuracy_After, label = 'Testing_Accuracy_After')
plt.xticks(range(len(Models)), Models, Rotation = 45)
plt.title('Testing Accuracy Behaviour')
plt.legend()
plt.show()