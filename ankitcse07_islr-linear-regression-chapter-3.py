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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
%matplotlib inline
## Read the dataset using pandas and see some entries
df = pd.read_csv('/kaggle/input/ISLR-Auto/Advertising.csv')
df.head()
## Check for different attributes
df.info()
## See for point summaries
df.describe()
## Check whether any column has null entries
df.isnull().sum()
### Find the total Advertising budget
df['Total_Advertising_Budget'] = df['TV'] + df['Radio'] + df['Newspaper']
## Visualize the total advertising budget vs Sales
sns.scatterplot(x='Total_Advertising_Budget', y='Sales', data=df)
## Check for correlation and draw the heat-map
corr = df.corr()
sns.heatmap(corr, vmax=1, vmin=-1, annot=True, cmap='plasma')
## We can also draw the pairplot for checking the relationship
sns.pairplot(df, diag_kind='kde')
df
## Prepare for linear regression
cData = df.drop(['Unnamed: 0', 'Total_Advertising_Budget', 'Sales'], axis=1)

## Identify X & Y
X = cData
y = df.Sales
## Split between train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)
print(X_train.shape)
print(X_test.shape)
## Prepare model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
for idx, col_name in enumerate(X_train.columns):
    print("Coefficeint of {} is {}".format(col_name, reg_model.coef_[idx]))
### Check the scores over training data
reg_model.score(X_train, y_train)
### Check the r2 score over test data. This is really awesome
reg_model.score(X_test, y_test)
y_predict = reg_model.predict(X_test)
y_predict
## Let's also see the residual plots for ei = y-y^ vs y^
y_predict_train = reg_model.predict(X_train)
E = y_train - y_predict_train
figure, ax = plt.subplots(1,1)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
sns.scatterplot(y_predict_train, E.values, ax=ax)
## From residual plots its clear that there seems to be some relationship between 
## residuals and predicted value. This means there is some level of non-linearity among predictors

### Let's also indentify the outliers too 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, y_predict_train)
rmse = np.sqrt(mse)


E[E/rmse >= 2]



## Let's create a new data frame with actual and predicted value
new_df = X_test.copy()
new_df['Sales_predict'] = y_predict
new_df['Sales_actual'] = y_test
new_df
## Let's plot the pairplot with this new data frame
## If you look from below graphs, test between each attribute and Sales_actual vs Sales_predict seems to be 
## quite close to each other
sns.pairplot(new_df, diag_kind='kde')
from sklearn.metrics import mean_squared_error, r2_score
## Find the root mean square error and r2_score
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)
print("Root mean square error = ", rmse)
print("R2 score", r2)
### let's add more interaction terms and see that our predict model get's better
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

## interaction_only states that only includes the interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True)

## Create the new training/test data with added interaction terms
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

## Again fit the linear regression model on such train data
poly_clf = linear_model.LinearRegression()
poly_clf.fit(X_train2, y_train)

#In sample (training) R^2 will always improve with the number of variables!
## See the effects of the interaction terms
print(poly_clf.score(X_train2, y_train))
print(poly_clf.score(X_test2, y_test))
### let's add higher degree polynomial terms and see that our predict model get's better
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2)

## Create the new training/test data with added interaction terms
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

## Again fit the linear regression model on such train data
poly_clf = linear_model.LinearRegression()
poly_clf.fit(X_train2, y_train)

#In sample (training) R^2 will always improve with the number of variables!
## See the effects of the interaction terms
print(poly_clf.score(X_train2, y_train))
print(poly_clf.score(X_test2, y_test))
## Let's also see the residual plots for ei = y-y^ vs y^
y_predict_train = poly_clf.predict(X_train2)
E = y_train - y_predict_train
figure, ax = plt.subplots(1,1)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
sns.scatterplot(y_predict_train, E.values, ax=ax)

## from below plot there doesn't seem be any relationship between Residuals and Fitted Values
## Let's also check distribution of the residuals
sns.distplot(E.values)
## Let's also build a OLS based model for regression
import statsmodels.api as sm
from statsmodels.api import add_constant
X = cData
Y = df.Sales
X2 = add_constant(X)
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.2, random_state=100)
lm = sm.OLS(Y_train, X_train)
lm2 = lm.fit()
lm2.summary()

## This will check the presence of multi-collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
