import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.metrics import mean_squared_error



import warnings

warnings.simplefilter('ignore')
path = '../input/'

housing = pd.read_csv(path + 'train.csv')

housing.head()
housing.shape
housing.isnull().sum()
housing = housing.loc[:, housing.isnull().sum() < 100]
housing.columns
correlations = housing.corr()['SalePrice']

correlations
correlations[(correlations > 0.5) | (correlations < -0.5)]
y = housing['SalePrice']



predictor_cols = ['OverallQual', 'YearBuilt', 

                  'YearRemodAdd', 'TotalBsmtSF', 

                  '1stFlrSF', 'GrLivArea', 

                  'FullBath', 'TotRmsAbvGrd', 

                  'GarageCars', 'GarageArea',

                 'Fireplaces', 'LotArea']



X = housing[predictor_cols]

X.head()
y.hist();
y = np.log1p(y)

y.hist();
X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                        random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)



print('R-squared score (training): {:.3f}'

     .format(linreg.score(X_train, y_train)))

print('R-squared score (validation): {:.3f}'

     .format(linreg.score(X_valid, y_valid)))
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

X_train_poly, X_valid_poly, y_train_poly, y_valid_poly = train_test_split(X_poly, y,

                                                   random_state = 0)
polyreg = LinearRegression().fit(X_train_poly, y_train_poly)



polyreg_train_score = polyreg.score(X_train_poly, y_train_poly)

polyreg_valid_score = polyreg.score(X_valid_poly, y_valid_poly)



print('R-squared score (training): {:.3f}'

     .format(polyreg_train_score))

print('R-squared score (validation): {:.3f}'

     .format(polyreg_valid_score))
polyreg_lasso = Lasso(alpha=100).fit(X_train_poly, y_train_poly)



print('R-squared score (training): {:.3f}'

     .format(polyreg_lasso.score(X_train_poly, y_train_poly)))

print('R-squared score (validation): {:.3f}'

     .format(polyreg_lasso.score(X_valid_poly, y_valid_poly)))
polyreg_ridge = Ridge(alpha=100).fit(X_train_poly, y_train_poly)



print('R-squared score (training): {:.3f}'

     .format(polyreg_ridge.score(X_train_poly, y_train_poly)))

print('R-squared score (validation): {:.3f}'

     .format(polyreg_ridge.score(X_valid_poly, y_valid_poly)))
def get_scores(reg):

    train_score = reg.score(X_train_poly, y_train_poly)

    valid_score = reg.score(X_valid_poly, y_valid_poly)

    return train_score, valid_score



def get_rmse(reg):

    y_pred_train = reg.predict(X_train_poly)

    train_rmse = np.sqrt(mean_squared_error(y_train_poly, y_pred_train))

    y_pred_valid = reg.predict(X_valid_poly)

    valid_rmse = np.sqrt(mean_squared_error(y_valid_poly, y_pred_valid))

    return train_rmse, valid_rmse



def ridge_validation_curve(alpha):

    reg = Ridge(alpha=alpha).fit(X_train_poly, y_train_poly)

    train_score, valid_score = get_scores(reg)

    train_rmse, valid_rmse = get_rmse(reg)  

    return train_score, valid_score, train_rmse, valid_rmse



def lasso_validation_curve(alpha):

    reg = Lasso(alpha=alpha).fit(X_train_poly, y_train_poly)

    train_score, valid_score = get_scores(reg)

    train_rmse, valid_rmse = get_rmse(reg)  

    return train_score, valid_score, train_rmse, valid_rmse



alphas = [0.1, 1, 5, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000]



scores_lasso = [lasso_validation_curve(alpha) for alpha in alphas]

scores_lasso_train = [s[0] for s in scores_lasso]

scores_lasso_valid = [s[1] for s in scores_lasso]

rmse_lasso_train = [s[2] for s in scores_lasso]

rmse_lasso_valid = [s[3] for s in scores_lasso]



scores_ridge = [ridge_validation_curve(alpha) for alpha in alphas]

scores_ridge_train = [s[0] for s in scores_ridge]

scores_ridge_valid = [s[1] for s in scores_ridge]

rmse_ridge_train = [s[2] for s in scores_ridge]

rmse_ridge_valid = [s[3] for s in scores_ridge]



scores_poly_train = [polyreg_train_score]*len(alphas)

scores_poly_valid = [polyreg_valid_score]*len(alphas)

y_pred_train = polyreg.predict(X_train_poly)

rmse_poly_train = [mean_squared_error(y_train_poly, y_pred_train)]*len(alphas)

y_pred_valid = polyreg.predict(X_valid_poly)

rmse_poly_valid = [mean_squared_error(y_valid_poly, y_pred_valid)]*len(alphas)
plt.figure(figsize=(10, 6));

plt.ylim([0.65, 0.9])

plt.xlabel('Regularization parameter (alpha)')

plt.ylabel('R-squared')

plt.title('R-squared scores as function of regularization')



plt.plot(alphas, scores_ridge_train, label='Poynomial with Ridge (training)')

plt.plot(alphas, scores_poly_train, label='Polynomial (training)')

plt.plot(alphas, scores_lasso_train, label='Poynomial with Lasso (training)')



plt.plot(alphas, scores_lasso_valid, label='Poynomial with Lasso (validation)')

plt.plot(alphas, scores_ridge_valid, label='Poynomial with Ridge (validation)')

plt.plot(alphas, scores_poly_valid, label='Polynomial (validation)')

plt.legend(loc=4);
plt.figure(figsize=(11, 6));

plt.ylim([0.012, 0.3])

plt.xlabel('Regularization parameter (alpha)')

plt.ylabel('Root Mean-squared Error(RMSE)')

plt.title('Root Mean-squared Error(RMSE) as a function of regularization')



plt.plot(alphas, rmse_lasso_valid, label='Poynomial with Lasso (validation)')

plt.plot(alphas, rmse_ridge_valid, label='Poynomial with Ridge (validation)')

plt.plot(alphas, rmse_poly_valid, label='Polynomial (validation)')



plt.plot(alphas, rmse_poly_train, label='Poynomial (training)')

plt.plot(alphas, rmse_lasso_train, label='Poynomial with Lasso (training)')

plt.plot(alphas, rmse_ridge_train, label='Poynomial with Ridge (training)')



plt.legend(loc=1);
housing_test = pd.read_csv(path + 'test.csv')

Id = housing_test['Id']

X_test = housing_test[predictor_cols]

X_test.head()
X_test.isnull().sum()
X_test = X_test.fillna(method='ffill')
reg = LinearRegression().fit(X_poly, y)
X_test_poly = poly.transform(X_test)

predictions = reg.predict(X_test_poly)

predictions[:10]
predictions = np.expm1(predictions) 

predictions[:10]
sample_submission = pd.read_csv(path + 'sample_submission.csv')

sample_submission.head()
submission = pd.DataFrame({'Id': Id,

                          'SalePrice': predictions})



submission.head()
submission.to_csv('my_submission.csv', index=False)