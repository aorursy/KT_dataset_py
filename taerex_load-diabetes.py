import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.datasets import load_diabetes
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
house_price = load_diabetes()
df = pd.DataFrame(house_price.data, columns=house_price.feature_names)
df['TEMP'] = house_price.target
df.head(10)
# standardize and train/test split
house_price.data = preprocessing.scale(house_price.data)
X_train, X_test, y_train, y_test = train_test_split(
    house_price.data, house_price.target, test_size=0.3, random_state=10)
ols_reg = LinearRegression()
ols_reg.fit(X_train, y_train)
ols_pred = ols_reg.predict(X_test)

pd.DataFrame({'variable': house_price.feature_names, 'estimate': ols_reg.coef_})
house_price.feature_names
# initialize
ridge_reg = Ridge(alpha=0)
ridge_reg.fit(X_train, y_train)
ridge_df = pd.DataFrame({'variable': house_price.feature_names, 'estimate': ridge_reg.coef_})
ridge_train_pred = []
ridge_test_pred = []

alphas = np.arange(0, 200, 1)

for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)
    var_name = 'estimate' + str(alpha)
    ridge_df[var_name] = ridge_reg.coef_
    # prediction
    ridge_train_pred.append(ridge_reg.predict(X_train))
    ridge_test_pred.append(ridge_reg.predict(X_test))
    
ridge_df = ridge_df.set_index('variable').T.rename_axis('estimate').reset_index()#.rename_axis(None, 1).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ridge_df.bmi, 'r', ridge_df.s1, 'g', ridge_df.age, 'b', ridge_df.sex, 'c', ridge_df.bp, 'y')
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression Trace", fontsize=16)
ax.legend(labels=['bmi','s1','age','sex','bp'])
ax.grid(True)
# initialize
lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train, y_train)
lasso_df = pd.DataFrame({'variable': house_price.feature_names, 'estimate': lasso_reg.coef_})
lasso_train_pred = []
lasso_test_pred = []

alphas = np.arange(0.01, 8.01, 0.04)

for alpha in alphas:
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train, y_train)
    var_name = 'estimate' + str(alpha)
    lasso_df[var_name] = lasso_reg.coef_
    # prediction
    lasso_train_pred.append(lasso_reg.predict(X_train))
    lasso_test_pred.append(lasso_reg.predict(X_test))

lasso_df = lasso_df.set_index('variable').T.rename_axis('estimate').reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ridge_df.bmi, 'r', ridge_df.s1, 'g', ridge_df.age, 'b', ridge_df.sex, 'c', ridge_df.bp, 'y')
ax.set_xlabel("Lambda")
ax.set_xticklabels(np.arange(-1, 10, 1))
ax.set_ylabel("Beta Estimate")
ax.set_title("Lasso Regression Trace", fontsize=16)
ax.legend(labels=['bmi','s1','age','sex','bp'])
ax.grid(True)
# R-squared of training set
ridge_r_squared_train = [r2_score(y_train, p) for p in ridge_train_pred]
lasso_r_squared_train = [r2_score(y_train, p) for p in lasso_train_pred]

# R-squared of test set
ridge_r_squared_test = [r2_score(y_test, p) for p in ridge_test_pred]
lasso_r_squared_test = [r2_score(y_test, p) for p in lasso_test_pred]

# ols for benchmark
ols_r_squared = r2_score(y_test, ols_pred)

# plot R-squared of training and test
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.rcParams['axes.grid'] = True

# training set and test set together
axes[0,0].plot(ridge_r_squared_train, 'b', ridge_r_squared_test, 'r')
axes[0,0].set_title("Ridge Regression R-squared", fontsize=16)
axes[0,0].set_ylabel("R-squared")

axes[0,1].plot(lasso_r_squared_train, 'b', lasso_r_squared_test, 'r')
axes[0,1].set_title("Lasso Regression R-squared", fontsize=16)

# test set curve
axes[1,0].plot(ridge_r_squared_test[:25], 'ro')
axes[1,0].axhline(y=ols_r_squared, color='g', linestyle='--')
axes[1,0].set_title("Ridge Test Set Zoom-in", fontsize=16)
axes[1,0].set_xlabel("Model Simplicity$\longrightarrow$")
axes[1,0].set_ylabel("R-squared")

axes[1,1].plot(lasso_r_squared_test[:25], 'ro')
axes[1,1].axhline(y=ols_r_squared, color='g', linestyle='--')
axes[1,1].set_title("Lasso Test Set Zoom-in", fontsize=16)
axes[1,1].set_xlabel("Model Simplicity$\longrightarrow$")
# MSE of training set
ridge_mse_train = [mean_squared_error(y_train, p) for p in ridge_train_pred]
lasso_mse_train = [mean_squared_error(y_train, p) for p in lasso_train_pred]

# MSE of test set
ridge_mse_test = [mean_squared_error(y_test, p) for p in ridge_test_pred]
lasso_mse_test = [mean_squared_error(y_test, p) for p in lasso_test_pred]

# ols mse for benchmark
ols_mse = mean_squared_error(y_test, ols_pred)

# plot MSE of training and test
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.rcParams['axes.grid'] = True

# training set and test set together
axes[0,0].plot(ridge_mse_train, 'b', ridge_mse_test, 'r')
axes[0,0].set_title("Ridge Regression MSE", fontsize=16)
axes[0,0].set_ylabel("MSE")

axes[0,1].plot(lasso_mse_train, 'b', lasso_mse_test, 'r')
axes[0,1].set_title("Lasso Regression MSE", fontsize=16)

# test set curve
axes[1,0].plot(ridge_mse_test[:25], 'ro')
axes[1,0].axhline(y=ols_mse, color='g', linestyle='--')
axes[1,0].set_title("Ridge Test Set MSE", fontsize=16)
axes[1,0].set_xlabel("Model Simplicity$\longrightarrow$")
axes[1,0].set_ylabel("MSE")

axes[1,1].plot(lasso_mse_test[:25], 'ro')
axes[1,1].axhline(y=ols_mse, color='g', linestyle='--')
axes[1,1].set_title("Lasso Test Set MSE", fontsize=16)
axes[1,1].set_xlabel("Model Simplicity$\longrightarrow$")
from sklearn.model_selection import GridSearchCV
# ols for comparison
print("OLS R-squared:", round(ols_r_squared, 4))
print("OLS MSE:", round(ols_mse, 4))
# parameter setup
param = {'alpha': np.arange(0.01, 10, 0.01)}

ridge_reg_grid = GridSearchCV(Ridge(), param)
ridge_reg_grid.fit(X_train, y_train)
ridge_grid_pred = ridge_reg_grid.predict(X_test)

print(ridge_reg_grid.best_estimator_)
print("\nR-Squared:", round(r2_score(y_test, ridge_grid_pred), 4))
print("MSE:", round(mean_squared_error(y_test, ridge_grid_pred), 4))
lasso_reg_grid = GridSearchCV(Lasso(), param)
lasso_reg_grid.fit(X_train, y_train)
lasso_grid_pred = lasso_reg_grid.predict(X_test)

print(lasso_reg_grid.best_estimator_)
print("\nR-Squared:", round(r2_score(y_test, lasso_grid_pred), 4))
print("MSE:", round(mean_squared_error(y_test, lasso_grid_pred), 4))
#!pip install yellowbrick
#from sklearn.linear_model import RidgeCV
from yellowbrick.regressor import AlphaSelection # visualization
from sklearn.linear_model import RidgeCV
alphas=np.arange(0.01, 10, 0.01)
#ridgeCV_reg = RidgeCV(alphas=alphas)
#ridgeCV_reg.fit(X_train, y_train)
visualizer = AlphaSelection(RidgeCV(alphas=alphas))

visualizer.fit(X_train, y_train)
g = visualizer.poof()

ridgeCV_pred = visualizer.predict(X_test)
print("R-squared:", round(r2_score(y_test, ridgeCV_pred), 4))
