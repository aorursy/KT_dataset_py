import pandas as pd
import os
pd.options.display.max_columns = None
house = pd.read_csv("../input/house-price/House_price.csv")
house
house.shape
Y=house['SalePrice']
house = house.drop(["Id", "SalePrice"], axis=1)
house_cont = house.select_dtypes(include='number')
house_cat = house.select_dtypes(exclude=["number"])
house_cat
#Let's check if the data set has any missing values.   #kaggle
house.columns[house.isnull().any()]
house_cont.loc[:, house_cont.isnull().any()]
house_cat.loc[:, house_cat.isnull().any()]
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(house_cont)
X = imp.transform(house_cont)
house_cont_imp = pd.DataFrame(X, columns=house_cont.columns)

house_cat.isnull().mean()
house_cat.columns[house_cat.isnull().mean() > 0.4]
house_cat = house_cat[house_cat.columns[house_cat.isnull().mean() < 0.4]]
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(house_cat)
X_cat = imp.transform(house_cat)
house_cat_imp = pd.DataFrame(X_cat, columns=house_cat.columns)

X_cat_nodummy = house_cat_imp.values
house_cat_imp = pd.get_dummies(house_cat_imp)
X_cat = house_cat_imp.values

from sklearn.model_selection import train_test_split
X_cont_train, X_cont_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=765)
X_cat_train, X_cat_test, Y_train, Y_test = train_test_split(X_cat, Y, test_size=0.25, random_state=765)
X_cat_nodummy_train, X_cat_nodummy_test, Y_train, Y_test = train_test_split(X_cat_nodummy, Y, test_size=0.25, random_state=765)
#from sklearn.preprocessing import KBinsDiscretizer
#est = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile')
#XX = est.fit(X_cont_train) ## Careful this transforms the whole X_train into bins ... shou;d do it only for vars with outliers
#XX.transform(X_cont_train)
#XX.transform(X_cont_test)

from sklearn.linear_model import LinearRegression
reg_cont = LinearRegression().fit(X_cont_train, Y_train)
reg_cat = LinearRegression().fit(X_cat_train, Y_train)
reg_cont.score(X_cont_train,Y_train) ## Return the R2 of the regression
reg_cat.score(X_cat_train,Y_train) ## Return the R2 of the regression
from sklearn.model_selection import cross_validate
cv_results = cross_validate(reg_cont, X_cont_train, Y_train, cv=10, scoring=('r2', 'neg_mean_squared_error'))
cv_results_cat = cross_validate(reg_cat, X_cat_train, Y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))
cv_results

mmse = np.mean(cv_results["test_r2"])
sdmse = np.std(cv_results["test_r2"])
cvmse = sdmse/mmse
print("(Continous) Average MSE=%s SD MSE=%s CV=%s" % (mmse, sdmse, cvmse))


mmse = np.mean(cv_results_cat["test_r2"])
sdmse = np.std(cv_results_cat["test_r2"])
cvmse = sdmse/mmse
print("(Categorical) Average MSE=%s SD MSE=%s CV=%s" % (mmse, sdmse, cvmse))
reg_cont.score(X_cont_test, Y_test)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler, RobustScaler

## Scale data - 


## Continous
## This is difficult here - the data have outliers, have different distributions
scaler = StandardScaler().fit(X_cont_train)
X_scaled_cont_train = scaler.transform(X_cont_train)
X_scaled_cont_test = scaler.transform(X_cont_test)


## Dummy
## Problems....but I will scale it anyway
scaler_cat = StandardScaler().fit(X_cat_train)
X_scaled_cat_train = scaler_cat.transform(X_cat_train)
X_scaled_cat_test = scaler_cat.transform(X_cat_test)

alphas = np.logspace(-.5, 3.5, 40)  ### exp(-.1) to exp(6.5) (by 40)

tuned_parms = [{'alpha': alphas}]
n_folds = 5

ridge = Ridge(random_state=12345, max_iter=130000)

clf = GridSearchCV(ridge, tuned_parms, cv=n_folds, refit=False)
clf.fit(X_scaled_cont_train, Y_train)

clf_cat = GridSearchCV(ridge, tuned_parms, cv=n_folds, refit=False)
clf_cat.fit(X_scaled_cat_train, Y_train)

## First plot (continous)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]]);

## Second plot (dummy)

scores = clf_cat.cv_results_['mean_test_score']
scores_std = clf_cat.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]]);
print("Best parameter alpha (continous):%s, associated scores=%s" % (clf.best_params_['alpha'], clf.best_score_))
print("Best parameter alpha (dummy):%s, associated scores=%s" % (clf_cat.best_params_['alpha'], clf_cat.best_score_))
ridge_cont = Ridge(alpha=clf.best_params_['alpha'])
ridge_cont.fit(X_scaled_cont_train, Y_train)

ridge_cat = Ridge(alpha=clf_cat.best_params_['alpha'])
ridge_cat.fit(X_scaled_cat_train, Y_train)

print("Test scores: (Continous only): %s - (Dummy only): %s" % (ridge_cont.score(X_scaled_cont_test, Y_test), ridge_cat.score(X_scaled_cat_test, Y_test)))
reg_cont.coef_
reg_scaled_cont = LinearRegression().fit(X_scaled_cont_train, Y_train)
reg_scaled_cont.coef_
reg_scaled_cont.score(X_scaled_cont_train, Y_train) - reg_cont.score(X_cont_train, Y_train)
ridge_cont.score(X_scaled_cont_test, Y_test)
ridge_cat.score(X_scaled_cat_test, Y_test)
alphas = np.logspace(-.5, 3.5, 40)  ### exp(-.1) to exp(6.5) (by 40)
tuned_parms = [{'alpha': alphas}]
n_folds = 5

lasso = Lasso(random_state=12345, max_iter=130000)

clf_lasso = GridSearchCV(lasso, tuned_parms, cv=n_folds, refit=False)
clf_lasso.fit(X_scaled_cont_train, Y_train)

clf_cat_lasso = GridSearchCV(lasso, tuned_parms, cv=n_folds, refit=False)
clf_cat_lasso.fit(X_scaled_cat_train, Y_train)

## First plot (continous)
scores = clf_lasso.cv_results_['mean_test_score']
scores_std = clf_lasso.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]]);

## Second plot (dummy)

scores = clf_cat_lasso.cv_results_['mean_test_score']
scores_std = clf_cat_lasso.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]]);
print("Best parameter alpha (continous):%s, associated scores=%s" % (clf_lasso.best_params_['alpha'], clf_lasso.best_score_))
print("Best parameter alpha (dummy):%s, associated scores=%s" % (clf_cat_lasso.best_params_['alpha'], clf_cat_lasso.best_score_))
lasso_cont = Lasso(alpha=clf_lasso.best_params_['alpha'])
lasso_cont.fit(X_scaled_cont_train, Y_train)

lasso_cat = Lasso(alpha=clf_cat_lasso.best_params_['alpha'])
lasso_cat.fit(X_scaled_cat_train, Y_train)

print("Test scores: (Continous only): %s - (Dummy only): %s" % (lasso_cont.score(X_scaled_cont_test, Y_test), lasso_cat.score(X_scaled_cat_test, Y_test)))
lasso_cont.score(X_scaled_cont_test, Y_test)
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(3)
scaler2 = StandardScaler()

X_scaled_poly_cont_train = scaler2.fit_transform(poly2.fit_transform(X_scaled_cont_train))  ## scaling twice!
X_scaled_poly_cont_test = scaler2.transform(poly2.transform(X_scaled_cont_test))  ## scaling twice!
np.shape(X_scaled_poly_cont_train)
reg_poly2_cont = LinearRegression().fit(X_scaled_poly_cont_train, Y_train)
reg_poly2_cont.score(X_scaled_poly_cont_train, Y_train)
reg_poly2_cont.score(X_scaled_poly_cont_test, Y_test)
alphas = np.logspace(2, 15, 40)  ## <- larger lambdas are allowed

tuned_parms = [{'alpha': alphas}]
n_folds = 5

ridge = Ridge(random_state=12345, max_iter=130000)
clf = GridSearchCV(ridge, tuned_parms, cv=n_folds, refit=False)
clf.fit(X_scaled_poly_cont_train, Y_train)   
## <= X_scaled_poly_cont_train

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]]);
clf.best_params_['alpha']
ridge_poly_cont = Ridge(alpha=clf.best_params_['alpha'])
ridge_poly_cont.fit(X_scaled_poly_cont_train, Y_train)
ridge_poly_cont.score(X_scaled_poly_cont_test, Y_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_scaled_cont_train,Y_train)
rf.score(X_scaled_cont_test,Y_test)
rf.fit(X_scaled_cat_train,Y_train)
rf.score(X_scaled_cat_test,Y_test)
