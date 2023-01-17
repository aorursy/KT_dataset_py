# Importing all the necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from scipy import stats

from scipy.stats import norm, skew

from sklearn.metrics import r2_score, mean_squared_error 

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from xgboost import XGBRegressor, plot_importance



import os

print(os.listdir("../input"))
#Reading the data

data = pd.read_csv('../input/train.csv')

data.describe()
#Checking for null values (in % terms)

(data.isnull().sum()/ len(data)).sort_values(ascending = False)[: 10]
data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis = 1, inplace = True)

data.head()
#Checking the distribution of the target feature 'SalePrice'

plt.figure(figsize = (12, 6))

sns.distplot(data['SalePrice'], fit = norm, color = 'olive')



#Obtaining the parameters for fitting the 'norm' curve

(µ, E) = norm.fit(data['SalePrice'])

print('\n µ = {:.2f} and E = {:.2f}'.format(µ, E))



plt.ylabel('Frequency')

plt.legend(['Normal dist. parameters(µ: {:.2f} && E: {:.2f})'.format(µ, E)], loc = 'best')

plt.title('SalePrice distribution')



#Obtaning the probaility plot

plt.figure(figsize = (12, 6))

stats.probplot(data['SalePrice'], plot = plt)
plt.figure(figsize = (15, 6))

sns.distplot(np.log1p(data['SalePrice']), fit = norm, color = 'teal')



#Obtaining the parameters for fitting the 'norm' curve

(µ, E) = norm.fit(np.log1p(data['SalePrice']))

print('µ = {:.2f} and E = {:.2f}'.format(µ, E))



plt.ylabel('Frequency')

plt.legend(['LogTransformed Normal dist. parameters(µ: {:.2f} && E: {:.2f})'.format(µ, E)], loc = 'best')

plt.title('LogTransformed SalePrice distribution')



#Obtaining the LogTransformed SalePrice probplot

plt.figure(figsize = (12, 6))

stats.probplot(np.log1p(data['SalePrice']), plot = plt)
data.describe()
corr = data.drop(['SalePrice'], axis = 1).corr()

plt.figure(figsize = (20, 20))

ax = sns.heatmap(corr, linewidths = 0.01, center = 0, cmap = sns.diverging_palette(20, 220, n = 200))

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

plt.suptitle('Pearson Correlation Heatmap')
#Checking the correlation of 'SalePrice' with other features

corr_SP = data.corr()['SalePrice'].sort_values(ascending = False)

#corr_SP = corr_SP.filter(lambda x: x > 0.6 or x < -0.6)

plt.figure(figsize = (20, 6))

corr_SP.drop('SalePrice').plot.bar()
sns.pairplot(data[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']])
#Log Transformation of skewed data

data['SalePrice'] = np.log1p(data['SalePrice'])

num_feats = data.dtypes[data.dtypes != 'object'].index



#Computing skewness and performing log transformation

skew_feats = data[num_feats].apply(lambda x: skew(x.dropna())) #computing skewness after removing null values, if any

skew_feats = skew_feats[skew_feats > 0.75] #selecting those numeric features that have a skewness > 0.75

skew_feats = skew_feats.index 

data[skew_feats] = np.log1p(data[skew_feats])



#Encoding of Categorical Features

data = pd.get_dummies(data)



#Filling up NaN values with the mean

data = data.fillna(data.mean())



#Data Standardization

#ss = StandardScaler()

#data = ss.fit_transform(data)



#Converting back to DataFrame

data = pd.DataFrame(data)

data.head()
#Splitting the data into training and test set

X, Y = data.drop(['SalePrice'], axis = 1), data['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 0)
cv_model = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], eps = 1e-3, n_alphas = 100, fit_intercept = True,

                      normalize = True, precompute = 'auto', max_iter = 2000, tol = 0.0001, cv = 6,

                      copy_X = True, verbose = 0, n_jobs = -1, positive = 'False', random_state = 0)

cv_model.fit(X_train, Y_train)

print('Optimal Alpha(α): %.8f' %cv_model.alpha_)

print('Optimal l1_ratio(ρ): %.3f' %cv_model.l1_ratio_)

print('Number of iterations %d' %cv_model.n_iter_)
Y_train_pred = cv_model.predict(X_train)

Y_test_pred = cv_model.predict(X_test)



print('Train r2 score:', r2_score(Y_train_pred, Y_train))

print('Test r2 score:', r2_score(Y_test_pred, Y_test))



train_mse = mean_squared_error(Y_train_pred, Y_train)

test_mse = mean_squared_error(Y_test_pred, Y_test)

train_rmse = np.sqrt(train_mse)

test_rmse = np.sqrt(test_mse)



print('Train RMSE is: %.4f' %train_rmse)

print('Test RMSE is: %.4f' %test_rmse)



#The RMSE here is actually RMSLE ( Root Mean Squared Logarithmic Error). Because we have taken the log of the actual values. 
features_imp = pd.Series(index = X_train.columns, data = np.abs(cv_model.coef_))

print('Total important features:',features_imp.shape)

n_selected_features = (features_imp > 0).sum()

print('Selected features: {0:d}, reduction of {1:2.2f}%'.format(n_selected_features,(1 - n_selected_features/ len(features_imp))*100))

features_imp.sort_values().tail(40).plot(kind = 'bar', figsize = (20, 6))
xgb_m1 = XGBRegressor()

xgb_m1.fit(X_train, Y_train)



#Predictions on Train & Test sets

Y_train_pred_xgb1 = xgb_m1.predict(X_train)

Y_test_pred_xgb1 = xgb_m1.predict(X_test)



#r2, mse & rmse scores for xgb_m1

print('XGBoost M1 Stats -->')

print('The r2 train score for XGBoost M1 is: %.2f' %r2_score(Y_train_pred_xgb1, Y_train))

print('The r2 test score for XGBoost M1 is: %.2f' %r2_score(Y_test_pred_xgb1, Y_test))

print('The mse train score for XGBoost M1 is: %.4f' %mean_squared_error(Y_train_pred_xgb1, Y_train))

print('The mse train score for XGBoost M1 is: %.4f' %mean_squared_error(Y_test_pred_xgb1, Y_test))

print('The mse train score for XGBoost M1 is: %.4f' %np.sqrt(mean_squared_error(Y_train_pred_xgb1, Y_train)))

print('The mse train score for XGBoost M1 is: %.4f' %np.sqrt(mean_squared_error(Y_test_pred_xgb1, Y_test)))
xgb_m2 = XGBRegressor(n_estimators = 1000)

xgb_m2.fit(X_train, Y_train, early_stopping_rounds = 5, eval_set = [(X_test, Y_test)], verbose = False)



#Predictions on Train & Test sets

Y_train_pred_xgb2 = xgb_m2.predict(X_train)

Y_test_pred_xgb2 = xgb_m2.predict(X_test)



#r2, mse & rmse scores for xgb_m2

print('XGBoost M2 Stats -->')

print('The r2 train score for XGBoost M2 is: %.2f' %r2_score(Y_train_pred_xgb2, Y_train))

print('The r2 test score for XGBoost M2 is: %.2f' %r2_score(Y_test_pred_xgb2, Y_test))

print('The mse train score for XGBoost M2 is: %.4f' %mean_squared_error(Y_train_pred_xgb2, Y_train))

print('The mse test score for XGBoost M2 is: %.4f' %mean_squared_error(Y_test_pred_xgb2, Y_test))

print('The rmse train score for XGBoost M2 is: %.4f' %np.sqrt(mean_squared_error(Y_train_pred_xgb2, Y_train)))

print('The rmse test score for XGBoost M2 is: %.4f' %np.sqrt(mean_squared_error(Y_test_pred_xgb2, Y_test)))
xgb_m3 = XGBRegressor(n_estimators = 2000, learning_rate = 0.5)

xgb_m3.fit(X_train, Y_train, early_stopping_rounds = 5, eval_set = [(X_test, Y_test)], verbose = False)



#Predictions on Train & Test sets

Y_train_pred_xgb3 = xgb_m3.predict(X_train)

Y_test_pred_xgb3 = xgb_m3.predict(X_test)



#r2, mse & rmse scores for xgb_m3

print('XGBoost M3 Stats -->')

print('The r2 train score for XGBoost M3 is: %.2f' %r2_score(Y_train_pred_xgb3, Y_train))

print('The r2 test score for XGBoost M3 is: %.2f' %r2_score(Y_test_pred_xgb3, Y_test))

print('The mse train score for XGBoost M3 is: %.4f' %mean_squared_error(Y_train_pred_xgb3, Y_train))

print('The mse test score for XGBoost M3 is: %.4f' %mean_squared_error(Y_test_pred_xgb3, Y_test))

print('The rmse train score for XGBoost M3 is: %.4f' %np.sqrt(mean_squared_error(Y_train_pred_xgb3, Y_train)))

print('The rmse test score for XGBoost M3 is: %.4f' %np.sqrt(mean_squared_error(Y_test_pred_xgb3, Y_test)))
from collections import OrderedDict

OrderedDict(sorted(xgb_m2.get_booster().get_fscore().items(), key = lambda x: x[1], reverse = True))
relevant_feats = list(dict((k, v) for k, v in xgb_m2.get_booster().get_fscore().items() if v >= 4).keys())

Xn = data[relevant_feats]

Yn = data['SalePrice']



X_train, X_test, Y_train, Y_test = train_test_split(Xn, Yn, test_size = 0.2, random_state = 0) #Splitting the data with the most relevant features

xgb_m4 = XGBRegressor(n_estimators = 1000) #Creating a new model on the lines of M2

xgb_m4.fit(X_train, Y_train, early_stopping_rounds = 5, eval_set = [(X_test, Y_test)], verbose = False)



#Predictions on Train & Test sets

Y_train_pred_xgb4 = xgb_m4.predict(X_train)

Y_test_pred_xgb4 = xgb_m4.predict(X_test)



#r2, mse & rmse scores for xgb_m4

print('XGBoost M4 Stats -->')

print('The r2 train score for XGBoost M4 is: %.2f' %r2_score(Y_train_pred_xgb4, Y_train))

print('The r2 test score for XGBoost M4 is: %.2f' %r2_score(Y_test_pred_xgb4, Y_test))

print('The mse train score for XGBoost M4 is: %.4f' %mean_squared_error(Y_train_pred_xgb4, Y_train))

print('The mse test score for XGBoost M4 is: %.4f' %mean_squared_error(Y_test_pred_xgb4, Y_test))

print('The rmse train score for XGBoost M4 is: %.4f' %np.sqrt(mean_squared_error(Y_train_pred_xgb4, Y_train)))

print('The rmse test score for XGBoost M4 is: %.4f' %np.sqrt(mean_squared_error(Y_test_pred_xgb4, Y_test)))