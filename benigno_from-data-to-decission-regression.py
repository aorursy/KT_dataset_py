import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Original data plotting
import seaborn as sns # Data plotting
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline
SEED = 52

from sklearn.datasets import load_boston # Dataset
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split # Utility for splitting
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score # For K-fold Cross Validation
from sklearn.linear_model import LinearRegression # Linear Regression Model
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures # To check for feature interaction


boston = load_boston() # It loads in dictionary format, where key 'data' and 'target' are the actual data.
# print(boston['DESCR']) # Information about dataset, omited here as it takes lots of space. Important information presented as text above.

X = pd.DataFrame(data = boston['data'],
                     columns = boston['feature_names'])
y = pd.Series(data = boston['target'], name = 'target')
Xy = pd.concat([X, y], axis = 1)
Xy.head()
Xy.describe()
X_train, X_test, y_train, y_test = train_test_split(Xy.iloc[:,:-1], Xy['target'], test_size=0.15, random_state=SEED)
columns_nochas = X_train.columns[X_train.columns != 'CHAS']
CHAS_train = X_train.CHAS.values
CHAS_test = X_test.CHAS.values
scaler = preprocessing.StandardScaler().fit(X_train.loc[:, columns_nochas])
X_train = pd.DataFrame(data = scaler.transform(X_train.loc[:, columns_nochas]), columns = columns_nochas)
X_train['CHAS'] = CHAS_train
X_test = pd.DataFrame(data = scaler.transform(X_test.loc[:, columns_nochas]), columns = columns_nochas)
X_test['CHAS'] = CHAS_test
# X_train.head()
g = sns.pairplot(Xy, height = 2, vars = ['LSTAT', 'PTRATIO', 'B', 'RM', 'target'],
                 kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}}) 
reg = LinearRegression(n_jobs = -1).fit(X_train, y_train)
print("R-square score on Train data: {0:.3f}".format(reg.score(X_train, y_train)))
print("R-square score on Test data: {0:.3f}".format(reg.score(X_test, y_test)))
coefs_lr = pd.DataFrame(data = reg.coef_, index = boston['feature_names'], columns=['Coeficients'])
coefs_lr.sort_values(by='Coeficients')
pol_feat = PolynomialFeatures(2).fit(X_train)
X_train_poly = pol_feat.transform(X_train)
X_test_poly = pol_feat.transform(X_test)
print("Number of newly created features =", X_train_poly.shape[1] - 13, 'for a total of',X_train_poly.shape[1],' features')
reg.fit(X_train_poly, y_train)
print("R-square score on Train Data: {0:.3f}".format(reg.score(X_train_poly, y_train)))
print("R-square score on Test Data: {0:.3f}".format(reg.score(X_test_poly, y_test)))

# Feature names taken from https://stackoverflow.com/questions/36728287/sklearn-preprocessing-polynomialfeatures-how-to-keep-column-names-headers-of
target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(X_train.columns,p) for p in pol_feat.powers_]]
coefs_lr = pd.DataFrame(data = reg.coef_, index = target_feature_names, columns=['Coeficients'])
# coefs_lr.sort_values(by='Coeficients') # Uncomment this line to check on the features and its coeficients.
scores = pd.DataFrame(data = { 'accuracy':[], 'std':[], 'max':[], 'min':[]})
for k in range(2, 16):
    score = cross_val_score(reg, X_train, y_train, cv=k, scoring=('r2'))
    scores.loc[k] = [score.mean(), score.std() * 2, score.mean() + score.std() * 2, score.mean() - score.std() * 2]
    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
scores[['accuracy','std']].sort_values(['accuracy'], ascending = False)[:3]
sns.lineplot(x = scores.index, y = 'accuracy', data=scores)
sns.lineplot(x = scores.index, y = 'max', data=scores, color = 'grey')
sns.lineplot(x = scores.index, y = 'min', data=scores, color = 'grey')
cv_results = cross_validate(reg, X_train, y_train, cv=3, return_train_score=True, scoring=('r2'))
df_cv_results = pd.DataFrame(data = { 'train_score':cv_results['train_score'], 'test_score':cv_results['test_score'] })
df_cv_results
scores = pd.DataFrame(data = { 'accuracy':[], 'std':[], 'max':[], 'min':[]})
for k in range(2, 16):
    score = cross_val_score(reg, X_train_poly, y_train, cv=k, scoring=('r2'))
    scores.loc[k] = [score.mean(), score.std() * 2, score.mean() + score.std() * 2, score.mean() - score.std() * 2]
    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
scores[['accuracy','std']].sort_values(['accuracy'], ascending = False)[:5]
sns.lineplot(x = scores.index, y = 'accuracy', data=scores)
sns.lineplot(x = scores.index, y = 'max', data=scores, color = 'grey')
sns.lineplot(x = scores.index, y = 'min', data=scores, color = 'grey')
# pol_feat = PolynomialFeatures(2).fit(X)
# X_poly = pol_feat.transform(X)
cv_results = cross_validate(reg, X_train, y_train, cv=4, return_train_score=True, scoring=('r2'))
df_cv_results = pd.DataFrame(data = { 'train_score':cv_results['train_score'], 'test_score':cv_results['test_score']})
df_cv_results
#Define the alpha values to test
alpha_lasso = [  1e-3, 0.0015, 0.0018, 0.01, 0.018, 0.015, 0.1]

for a in alpha_lasso:
    lassoreg = Lasso(alpha = a, normalize=True, max_iter=1e5)
    lassoreg.fit(X_train, y_train)
    print("R-square score on Train data for alpha = {0}: {1:.5f}".format(a, lassoreg.score(X_train, y_train)))
    print("R-square score on Test data for alpha = {0}: {1:.5f}".format(a, lassoreg.score(X_test, y_test)))
from sklearn.model_selection import KFold

kf = KFold(n_splits = 2)

alphas = [ 1e-15, 1e-12, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 0.0015, 0.00175, 0.0018, 0.00185, 0.00186, 0.00187, 0.0019, 0.002, 0.01, 0.018, 0.015, 0.1, 1, 5]

e_alphas = list()
e_alphas_r = list()  #holds average r2 error
for alpha in alphas:
    lasso = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    err = list()
    err_2 = list()
    for tr_idx, tt_idx in kf.split(X_train):
        X_tr , X_tt = X_train.iloc[tr_idx], X_train.iloc[tt_idx]
        y_tr, y_tt = y_train.iloc[tr_idx], y_train.iloc[tt_idx]
        lasso.fit(X_tr, y_tr)
        y_hat = lasso.predict(X_tt)
        
        # returns the coefficient of determination (R^2 value)
        err_2.append(lasso.score(X_tt,y_tt))
        
        # returns MSE
        err.append(np.average((y_hat - y_tt)**2))
        
    e_alphas.append(np.average(err))
    e_alphas_r.append(np.average(err_2))

## print out the alpha that gives the minimum error
print('the minimum value of error is ', e_alphas[e_alphas.index(min(e_alphas))])
print('the minimizer is ',  alphas[e_alphas.index(min(e_alphas))])

##  <<< plotting alphas against error >>>
plt.figsize = (15,10)
fig = plt.figure()     
ax = fig.add_subplot(111)
ax.plot(alphas, e_alphas, 'b-', label = 'MSE')
ax.plot(alphas, e_alphas_r, 'g--', label = '')
ax.set_xlabel("alpha")
plt.show()
lassoreg = Lasso(alpha = 0.00185, normalize=True, max_iter=1e5)
lassoreg.fit(X_train, y_train)
print("R-square score on Train data for alpha = {0}: {1:.5f}".format(a, lassoreg.score(X_train, y_train)))
print("R-square score on Test data for alpha = {0}: {1:.5f}".format(a, lassoreg.score(X_test, y_test)))
coefs_lasso = pd.DataFrame(data = lassoreg.coef_, index = X_train.columns, columns=['Coeficients'])
coefs_lasso.sort_values(by='Coeficients')
from sklearn.model_selection import KFold

kf = KFold(n_splits = 4)

alphas = [ 1e-5, 1e-4, 4.5e-4, 4.7e-4, 5e-4, 5.5e-4, 1e-3, 0.01, 0.018, 0.015, 0.1]

e_alphas = list()
e_alphas_r = list()  #holds average r2 error
for alpha in alphas:
    lasso = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    err = list()
    err_2 = list()
    for tr_idx, tt_idx in kf.split(X_train_poly):
        X_tr , X_tt = X_train_poly[tr_idx], X_train_poly[tt_idx]
        y_tr, y_tt = y_train.iloc[tr_idx], y_train.iloc[tt_idx]
        lasso.fit(X_tr, y_tr)
        y_hat = lasso.predict(X_tt)
        
        # returns the coefficient of determination (R^2 value)
        err_2.append(lasso.score(X_tt,y_tt))
        
        # returns MSE
        err.append(np.average((y_hat - y_tt)**2))
        
    e_alphas.append(np.average(err))
    e_alphas_r.append(np.average(err_2))

## print out the alpha that gives the minimum error
print('the minimum value of error is ', e_alphas[e_alphas.index(min(e_alphas))])
print('the minimizer is ',  alphas[e_alphas.index(min(e_alphas))])
alpha = 0.00047
lassoreg = Lasso(alpha = alpha, normalize=True, max_iter=1e5)
lassoreg.fit(X_train_poly, y_train)
print("R-square score on Train data {1:.5f}".format(a, lassoreg.score(X_train_poly, y_train)))
print("R-square score on Test data {1:.5f}".format(a, lassoreg.score(X_test_poly, y_test)))
print("Number of coeficients with alpha = {0} not zeroed by Lasso {1} out of 105".format(alpha, len(lassoreg.coef_[lassoreg.coef_ != 0])))

# Feature names taken from https://stackoverflow.com/questions/36728287/sklearn-preprocessing-polynomialfeatures-how-to-keep-column-names-headers-of
target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(X_train.columns,p) for p in pol_feat.powers_]]
coefs_lasso = pd.DataFrame(data = lassoreg.coef_, index = target_feature_names, columns=['Coeficients'])
#coefs_lasso.sort_values(by='Coeficients') # Uncomment this line to check on the features and its coeficients.
lass2 = LassoCV(cv=4, random_state=SEED, normalize = True).fit(X_train_poly, y_train)
print("R-square score on Train data {1:.5f}".format(a, lass2.score(X_train_poly, y_train)))
print("R-square score on Test data {1:.5f}".format(a, lass2.score(X_test_poly, y_test)))
print("Alpha used =",lass2.alpha_)
ridge = RidgeCV(cv = 4, normalize = True).fit(X_train_poly[:,np.where(lassoreg.coef_ != 0)].squeeze(), y_train)
print("R-square score on Train data {1:.5f}".format(a, ridge.score(X_train_poly[:,np.where(lassoreg.coef_ != 0)].squeeze(), y_train)))
print("R-square score on Test data {1:.5f}".format(a, ridge.score(X_test_poly[:,np.where(lassoreg.coef_ != 0)].squeeze(), y_test)))
print("Alpha used =",ridge.alpha_)