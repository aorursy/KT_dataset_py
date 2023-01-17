from mlxtend.regressor import StackingRegressor

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.simplefilter('ignore')
##Generating sample data

np.random.seed(1)

X = np.sort(5*np.random.rand(40,1), axis = 0)

y = np.sin(X).ravel()

y[::5] =  3 *(0.5 - np.random.rand(8))
##Initializing models

lr = LinearRegression()

svr_lin = SVR(kernel= 'linear')

ridge = Ridge(random_state =1)

svr_rbf = SVR(kernel ='rbf')



strreg = StackingRegressor(regressors = [lr, svr_lin, ridge],

                          meta_regressor = svr_rbf)



strreg.fit(X,y)

strreg.predict(X)



print("Mean squared error : %0.4f" %(np.mean((strreg.predict(X)-y)**2)))

print("Variance score : %0.4f" % strreg.score(X,y))



with plt.style.context('seaborn-whitegrid'):

    plt.scatter(X,y, c='lightgray')

    plt.plot(X, strreg.predict(X), c='darkgreen', lw = 2)

    

plt.show()
## stacking regressor with grid search



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
lasso = Lasso(random_state=1)

ridge = Ridge(random_state=1)

lr = LinearRegression()

svr_lin = SVR(kernel ='linear')

svr_rbf= SVR(kernel = 'rbf')

stregr = StackingRegressor(regressors=[svr_lin, lr, lasso, ridge],

                          meta_regressor = svr_rbf)



params = {'lasso__alpha' : [0.1,1.0,10.0],

         'ridge__alpha':[0.1,1.0,10.0],

         'svr__C':[0.1,1.0,10.0],

         'meta_regressor__C':[0.1,1.0,10.0,100.0],

         'meta_regressor__gamma':[0.1,1.0,10.0]}



grid = GridSearchCV(estimator = stregr, param_grid = params, cv = 5, refit = True)



grid.fit(X,y)

print("Best score is %f using best params %s" %(grid.best_score_, grid.best_params_))

cv_keys = ('mean_test_score','std_test_score', 'params')



for r, _ in enumerate(grid.cv_results_['mean_test_score']):

    print("%0.3f +/- %0.3f - %s" %(grid.cv_results_[cv_keys[0]][r],

                                   grid.cv_results_[cv_keys[1]][r],

                                   grid.cv_results_[cv_keys[2]][r]))

    

    if r>10:

        break

    print("....")

    

print("Best parameters : ",grid.best_params_)

print("Best score: " , grid.best_score_)
print('Mean squared error: ', np.mean((grid.predict(X)-y)**2))

print('Variance score:' , grid.score(X,y))
with plt.style.context('seaborn-whitegrid'):

    plt.scatter(X,y, c='lightgray')

    plt.plot(X, grid.predict(X), c='darkgreen', lw = 2)



plt.show()