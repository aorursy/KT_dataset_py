# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("../input/Advertising.csv", index_col = 0)
data.head()
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
sns.pairplot(data=data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='reg')
sns.distplot(y, hist=True, bins=20)
y_log = np.log(y)
sns.distplot(y_log, hist=True)
X.hist(bins=40)
from scipy.stats import skew
data_num_skew = X.apply(lambda x: skew(x.dropna()))
data_num_skewed = data_num_skew[data_num_skew > .75]

print(data_num_skew)
print(data_num_skewed)
import numpy as np
# apply log + 1 transformation for all numeric features with skewnes over .75
X[data_num_skewed.index] = np.log1p(X[data_num_skewed.index])
X.hist(bins=40)
corr_df = X.corr(method = 'pearson')
print(corr_df)

sns.heatmap(corr_df, vmax=1.0, vmin=-1.0, annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df["features"] = X.columns
vif_df["VIF Factor"] = [vif(X.values, i) for i in range(X.shape[1])]
vif_df.round(2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)
print(list(zip(X.columns, lm.coef_)))
X1 = 50
X2 = 50
X3 = 50
y_pred = 3.56702251819 + (0.04299773*X1) + (0.19279679*X2) + (-0.08103522*X3)
print(y_pred)
y_pred = lm.predict(X_test)
print(y_pred)
new_df = pd.DataFrame()
new_df = X_test

new_df['Actual Values'] = y_test
new_df['Predicted Values'] = y_pred
print(new_df.head())
from sklearn.metrics import r2_score, mean_squared_error
r2score = r2_score(y_test, y_pred)
print(r2score)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
print(min(y))
print(max(y))
import statsmodels.formula.api as sm
lm_model = sm.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

print(lm_model.params)
print(lm_model.summary())
y_pred_new = lm_model.predict(X_test)
r2score = r2_score(y_test, y_pred_new)
print(r2score)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_new))
print(rmse)
lm_model = sm.ols(formula='Sales ~ TV + Radio', data=data).fit()

print(lm_model.params)
print(lm_model.summary())

y_pred_new = lm_model.predict(X_test)

r2score = r2_score(y_test, y_pred_new)
print(r2score)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_new))
print(rmse)
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

# fitted values (need a constant term for intercept)
model_fitted_y = lm_model.fittedvalues

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Sales', data=data, lowess=True)

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plt.show()
res = lm_model.resid
import statsmodels.api as stm
import scipy.stats as stats
fig = stm.qqplot(res, fit=True, line='45')
plt.title('Normal Q-Q')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Standardized Residuals');
plt.show()
# normalized residuals
model_norm_residuals = lm_model.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, lowess=True)

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');



