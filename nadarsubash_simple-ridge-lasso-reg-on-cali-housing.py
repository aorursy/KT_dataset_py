import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
data = pd.read_csv("../input/california-housing-value/housing.csv")

data.head()
data.info()
data.isnull().sum()
data['total_bedrooms'].fillna(data['total_bedrooms'].mean(skipna=True),inplace=True)
data.isnull().sum()
data1 = pd.get_dummies(data,drop_first=True)

data1.head()
set(data['ocean_proximity'])
X = data1.drop(['median_house_value'],1)

y = data1['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train1 = scale(X_train)

X_test1 = scale(X_test)
X_train1[0]
classifier_LR = LinearRegression()

classifier_LR.fit(X_train1, y_train)

pred_LR = classifier_LR.predict(X_test1)

RMSE_LR = mean_squared_error(y_test, pred_LR)*0.5

print("RMSE for Linear Regression = ", RMSE_LR)
#Define random Alpha values

alphas = 10 ** np.linspace(10, -2, 100)*0.5 



#Use ridge regression cross validation to identify most optimal Alpha

ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)

ridgecv.fit(X_train1, y_train)

alpha_r = ridgecv.alpha_

print("Optimal Alpha identified for ridge is {}".format(alpha_r))
# Apply Ridge regression to predict median house value

ridge = Ridge(alpha=alpha_r, normalize=True)

ridge.fit(X_train1, y_train)

pred = ridge.predict(X_test1)

RMSE_RR = mean_squared_error(y_test, pred)*0.5

print("RMSE for Ridge Regression = ", RMSE_RR)
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)

lassocv.fit(X_train1, y_train)

lasso_score = lassocv.score(X_train1, y_train)

lasso_alpha = lassocv.alpha_

print("Lasso score is {} & Lasso alpha is {}".format(lasso_score, lasso_alpha))
lasso = Lasso(max_iter = 10000, normalize = True)

lasso.set_params(alpha=lassocv.alpha_)

lasso.fit(X_train1, y_train)

RMSE_LS = mean_squared_error(y_test, lasso.predict(X_test1))

print("RMSE for Lasso Regression = {}".format(RMSE_LS))
print("RMSE for Linear Regression = {}".format(RMSE_LR))

print("RMSE for Ridge Regression = {}".format(RMSE_RR))

print("RMSE for Lasso Regression = {}".format(RMSE_LS))