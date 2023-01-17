import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

import warnings

warnings.filterwarnings('ignore')

import os

import matplotlib.pyplot as plt
ad_data = pd.read_csv('../input/Advertising.csv',index_col='Unnamed: 0')
ad_data.info()
ad_data.describe()
p = sns.pairplot(ad_data)
# visualize the relationship between the features and the response using scatterplots

p = sns.pairplot(ad_data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)
x = ad_data.drop(["Sales"],axis=1)

y = ad_data.Sales
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import linear_model



regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)

y_pred = regr.predict(X_train)
print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))
residuals = y_train.values-y_pred

mean_residuals = np.mean(residuals)

print("Mean of Residuals {}".format(mean_residuals))
p = sns.scatterplot(y_pred,residuals)

plt.xlabel('y_pred/predicted values')

plt.ylabel('Residuals')

plt.ylim(-10,10)

plt.xlim(0,26)

p = sns.lineplot([0,26],[0,0],color='blue')

p = plt.title('Residuals vs fitted values plot for homoscedasticity check')
import statsmodels.stats.api as sms

from statsmodels.compat import lzip

name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(residuals, X_train)

lzip(name, test)
from scipy.stats import bartlett

test = bartlett( X_train,residuals)

print(test)
p = sns.distplot(residuals,kde=True)

p = plt.title('Normality of error terms/residuals')
plt.figure(figsize=(10,5))

p = sns.lineplot(y_pred,residuals,marker='o',color='blue')

plt.xlabel('y_pred/predicted values')

plt.ylabel('Residuals')

plt.ylim(-10,10)

plt.xlim(0,26)

p = sns.lineplot([0,26],[0,0],color='red')

p = plt.title('Residuals vs fitted values plot for autocorrelation check')
from statsmodels.stats import diagnostic as diag

min(diag.acorr_ljungbox(residuals , lags = 40)[1])
import statsmodels.api as sm
# autocorrelation

sm.graphics.tsa.plot_acf(residuals, lags=40)

plt.show()
# partial autocorrelation

sm.graphics.tsa.plot_pacf(residuals, lags=40)

plt.show()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(ad_data.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
from sklearn.tree import DecisionTreeRegressor



dec_tree = DecisionTreeRegressor(random_state=0)

dec_tree.fit(X_train,y_train)

dec_tree_y_pred = dec_tree.predict(X_train)

print("Accuracy: {}".format(dec_tree.score(X_train,y_train)))

print("R squared: {}".format(r2_score(y_true=y_train,y_pred=dec_tree_y_pred)))
from sklearn.ensemble import RandomForestRegressor



rf_tree = RandomForestRegressor(random_state=0)

rf_tree.fit(X_train,y_train)

rf_tree_y_pred = rf_tree.predict(X_train)

print("Accuracy: {}".format(rf_tree.score(X_train,y_train)))

print("R squared: {}".format(r2_score(y_true=y_train,y_pred=rf_tree_y_pred)))
from sklearn.svm import SVR



svr = SVR()

svr.fit(X_train,y_train)

svr_y_pred = svr.predict(X_train)

print("Accuracy: {}".format(svr.score(X_train,y_train)))

print("R squared: {}".format(r2_score(y_true=y_train,y_pred=svr_y_pred)))