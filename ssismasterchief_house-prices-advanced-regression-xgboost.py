import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



from scipy import stats

from scipy.stats import norm, skew



from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNetCV, ElasticNet

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold



from xgboost import XGBRegressor, plot_importance 
train_df = pd.read_csv("../input/train.csv")

train_df.head()
train_df.describe()
train_df.shape
# checking for features having the most occurrances of NULL/NaN values

# displaying top 10

(train_df.isnull().sum() / len(train_df)).sort_values(ascending=False)[:10]
# removing the features with more than 80% NULL/NaN values

# removing the feature 'Id', cannot be used for training model

train_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id'], axis=1, inplace=True)
train_df.shape
sb.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['SalePrice'])

#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('Sale Price distribution')

#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)

plt.show();
# visualisation show that data for 'SalePrice' is right skewed

# normalising 'SalePrice'

sb.distplot(np.log1p(train_df['SalePrice']) , fit=norm);

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(np.log1p(train_df['SalePrice']))

#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('log(Sale Price+1) distribution')

#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(np.log1p(train_df['SalePrice']), plot=plt)

plt.show();
# finding correlation between all features excluding 'SalePrice'

pd.set_option('precision',2)

plt.figure(figsize=(10, 8))

sb.heatmap(train_df.drop(['SalePrice'],axis=1).corr(), square=True)

plt.suptitle("Pearson Correlation Heatmap")

plt.show();
# visualising correlation between 'SalePrice' and all the other features

corr_with_sale_price = train_df.corr()["SalePrice"].sort_values(ascending=False)

plt.figure(figsize=(14,6))

corr_with_sale_price.drop("SalePrice").plot.bar()

plt.show();
'''

top features correlating to 'SalePrice':

1. OverallQual

2. GrLivArea

3. GarageCars



plotting paiplot for performing EDA

'''

sb.pairplot(train_df[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']])

plt.show();
# data preprocessing on remaining dataset



train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#log transform skewed numeric features:

numeric_feats = train_df.dtypes[train_df.dtypes != "object"].index



skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



train_df[skewed_feats] = np.log1p(train_df[skewed_feats])

train_df = pd.get_dummies(train_df)

train_df = train_df.fillna(train_df.mean())



X, y = train_df.drop(['SalePrice'], axis = 1), train_df['SalePrice']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape
cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=1e-3, n_alphas=100, fit_intercept=True, 

                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=6, 

                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=0)

               

cv_model.fit(X_train, y_train)

print('Optimal alpha: %.8f'%cv_model.alpha_)

print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)

print('Number of iterations %d'%cv_model.n_iter_)
y_train_pred = cv_model.predict(X_train)

y_pred = cv_model.predict(X_test)

print('Train r2 score: ', r2_score(y_train_pred, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred))

train_mse = mean_squared_error(y_train_pred, y_train)

test_mse = mean_squared_error(y_pred, y_test)

train_rmse = np.sqrt(train_mse)

test_rmse = np.sqrt(test_mse)

print('Train RMSE: %.4f' % train_rmse)

print('Test RMSE: %.4f' % test_rmse)
feature_importance = pd.Series(index = X_train.columns, data = np.abs(cv_model.coef_))

n_selected_features = (feature_importance>0).sum()

print('{0:d} features, reduction of {1:2.2f}%'.format(

    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (12,5));
xgb_model1 = XGBRegressor()

xgb_model1.fit(X_train, y_train, verbose=False)

y_train_pred1 = xgb_model1.predict(X_train)

y_pred1 = xgb_model1.predict(X_test)



print('Train r2 score: ', r2_score(y_train_pred1, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred1))

train_mse1 = mean_squared_error(y_train_pred1, y_train)

test_mse1 = mean_squared_error(y_pred1, y_test)

train_rmse1 = np.sqrt(train_mse1)

test_rmse1 = np.sqrt(test_mse1)

print('Train RMSE: %.4f' % train_rmse1)

print('Test RMSE: %.4f' % test_rmse1)
xgb_model2 = XGBRegressor(n_estimators=1000)

xgb_model2.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], verbose=False)

y_train_pred2 = xgb_model2.predict(X_train)

y_pred2 = xgb_model2.predict(X_test)



print('Train r2 score: ', r2_score(y_train_pred2, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred2))

train_mse2 = mean_squared_error(y_train_pred2, y_train)

test_mse2 = mean_squared_error(y_pred2, y_test)

train_rmse2 = np.sqrt(train_mse2)

test_rmse2 = np.sqrt(test_mse2)

print('Train RMSE: %.4f' % train_rmse2)

print('Test RMSE: %.4f' % test_rmse2)
xgb_model3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_model3.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], verbose=False)

y_train_pred3 = xgb_model3.predict(X_train)

y_pred3 = xgb_model3.predict(X_test)



print('Train r2 score: ', r2_score(y_train_pred3, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred3))

train_mse3 = mean_squared_error(y_train_pred3, y_train)

test_mse3 = mean_squared_error(y_pred3, y_test)

train_rmse3 = np.sqrt(train_mse3)

test_rmse3 = np.sqrt(test_mse3)

print('Train RMSE: %.4f' % train_rmse3)

print('Test RMSE: %.4f' % test_rmse3)
from collections import OrderedDict

OrderedDict(sorted(xgb_model2.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
most_relevant_features= list(dict((k, v) for k, v in xgb_model2.get_booster().get_fscore().items() if v >= 4).keys())

train_x=train_df[most_relevant_features]

train_y=train_df['SalePrice']

X_train, X_test, y_train, y_test  = train_test_split(train_x, train_y, test_size = 0.2, random_state = 0)

xgb_model4 = XGBRegressor(n_estimators=1000)

xgb_model4.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], verbose=False)

y_train_pred4 = xgb_model4.predict(X_train)

y_pred4 = xgb_model4.predict(X_test)



print('Train r2 score: ', r2_score(y_train_pred4, y_train))

print('Test r2 score: ', r2_score(y_test, y_pred4))

train_mse4 = mean_squared_error(y_train_pred4, y_train)

test_mse4 = mean_squared_error(y_pred4, y_test)

train_rmse4 = np.sqrt(train_mse4)

test_rmse4 = np.sqrt(test_mse4)

print('Train RMSE: %.4f' % train_rmse4)

print('Test RMSE: %.4f' % test_rmse4)