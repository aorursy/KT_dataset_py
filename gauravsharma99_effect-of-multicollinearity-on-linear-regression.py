import scipy

import numpy as np



from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
# cook data

np.random.seed(42)



x1 = np.random.randn(100)

y = 2 * x1 + 10  # one unit change in x1 changes 2 units of y in same direction



x1 = x1 + 0.25 * np.random.randn(100)

x2 = -x1 # one unit change in x2 changes 2 units of y but in opposite direction
# plot cooked data

fig = pyplot.figure(1, (16,4))



ax1 = pyplot.subplot(1,3,1)

ax1.scatter(x1, y)

ax1.set_title('x1 v/s y')



ax2 = pyplot.subplot(1,3,2)

ax2.scatter(x2, y)

ax2.set_title('x2 v/s y')



ax3 = pyplot.subplot(1,3,3)

ax3.scatter(x1, x2)

ax3.set_title('x1 v/s x2')



pyplot.tight_layout()

pyplot.show()
# quantitative look at correlations

print(scipy.stats.pearsonr(x1, y))

print(scipy.stats.pearsonr(x2, y))

print(scipy.stats.pearsonr(x1, x2)) # -1 value means perfect correlation which we expect
X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))

X.shape
lr = LinearRegression().fit(X[:,:1], y) # fit x1 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,1:], y) # fit x2 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X, y) # fit both x1 and x2

lr.coef_, lr.intercept_
# cook data

np.random.seed(42)

x1 = np.random.randn(100)

y1 = 2 * x1 + 4



np.random.seed(142)

x2 = np.random.randn(100)

y2 = 2 * x2 + 4



y = y1 + y2
# plot cooked data

fig = pyplot.figure(1, (16,4))



ax1 = pyplot.subplot(1,3,1)

ax1.scatter(x1, y)

ax1.set_title('x1 v/s y')



ax2 = pyplot.subplot(1,3,2)

ax2.scatter(x2, y)

ax2.set_title('x2 v/s y')



ax3 = pyplot.subplot(1,3,3)

ax3.scatter(x1, x2)

ax3.set_title('x1 v/s x2')



pyplot.tight_layout()

pyplot.show()
# quantitative look at correlations

print(scipy.stats.pearsonr(x1, y))

print(scipy.stats.pearsonr(x2, y))

print(scipy.stats.pearsonr(x1, x2))
X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))

X.shape
lr = LinearRegression().fit(X[:,:1], y) # fit x1 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,1:], y) # fit x2 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X, y) # fit both x1 and x2

lr.coef_, lr.intercept_
# cook data

np.random.seed(42)

x1 = np.random.randn(100)

y1 = 2 * x1 + 4 # one unit change in x1 changes 2 units of y in same direction



np.random.seed(142)

x2 = np.random.randn(100)

y2 = 2 * x2 + 4 # one unit change in x2 changes 2 units of y in same direction



x3 = -x1 # one unit change in x3 changes 2 units of y in opposite direction



y = y1 + y2
# plot cooked data

fig = pyplot.figure(1, (18,8))



ax = pyplot.subplot(2,3,1)

pyplot.scatter(x1, y)

ax.set_title('x1 v/s y')



ax = pyplot.subplot(2,3,2)

pyplot.scatter(x2, y)

ax.set_title('x2 v/s y')



ax = pyplot.subplot(2,3,3)

pyplot.scatter(x3, y)

ax.set_title('x3 v/s y')



ax = pyplot.subplot(2,3,4)

pyplot.scatter(x1, x2)

ax.set_title('x1 v/s x2')



ax = pyplot.subplot(2,3,5)

pyplot.scatter(x1, x3)

ax.set_title('x1 v/s x3')



ax = pyplot.subplot(2,3,6)

pyplot.scatter(x2, x3)

ax.set_title('x2 v/s x3')



pyplot.tight_layout()

pyplot.show()
# quantitative look at correlations

print(scipy.stats.pearsonr(x1, y))

print(scipy.stats.pearsonr(x2, y))

print(scipy.stats.pearsonr(x3, y))

print(scipy.stats.pearsonr(x1, x2))

print(scipy.stats.pearsonr(x1, x3))

print(scipy.stats.pearsonr(x2, x3))
X = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1)))

X.shape
lr = LinearRegression().fit(X[:,:1], y) # fit x1 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,1:2], y) # fit x2 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,2:], y) # fit x3 only

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,:2], y) # fit x1 & x2

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,[0,2]], y) # fit x1 & x3

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X[:,1:], y) # fit x2 & x3

lr.coef_, lr.intercept_
lr = LinearRegression().fit(X, y) # fit all x1,x2,x3

lr.coef_, lr.intercept_
# we cook the same experiment_1 data

np.random.seed(42)



x1 = np.random.randn(100)

y = 2 * x1 + 10  # one unit change in x1 changes 2 units of y in same direction



x1 = x1 + 0.25 * np.random.randn(100)

x2 = -x1 # one unit change in x2 changes 2 units of y in opposite direction



X = np.hstack((x1.reshape(-1,1),x2.reshape(-1,1)))

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

X_train.shape, X_test.shape
lr = LinearRegression().fit(X_train[:, :1], y_train) # fit x1 only



print('train score: ', lr.score(X_train[:, :1], y_train))

print('test score: ', lr.score(X_test[:, :1], y_test))

print('estimated params: ', *lr.coef_, lr.intercept_)
lr = LinearRegression().fit(X_train, y_train) # fit both x1 & x2



print('train score: ', lr.score(X_train, y_train))

print('test score: ', lr.score(X_test, y_test))

print('estimated params: ', *lr.coef_, lr.intercept_)
# we cook the same exp3 data

np.random.seed(42)

x1 = np.random.randn(100)

y1 = 2 * x1 + 4 # one unit change in x1 changes 2 units of y in same direction



np.random.seed(142)

x2 = np.random.randn(100)

y2 = 2 * x2 + 4 # one unit change in x2 changes 2 units of y in same direction



x3 = -x1 # one unit change in x3 changes 2 units of y in opposite direction



y = y1 + y2



X = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1)))

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

X_train.shape, X_test.shape
lr = LinearRegression().fit(X_train[:, :2], y_train) # fit x1 & x2 only



print('train score: ', lr.score(X_train[:, :2], y_train))

print('test score: ', lr.score(X_test[:, :2], y_test))

print('estimated params: ', *lr.coef_, lr.intercept_)
lr = LinearRegression().fit(X_train, y_train) # fit all x1,x2 & x3



print('train score: ', lr.score(X_train, y_train))

print('test score: ', lr.score(X_test, y_test))

print('estimated params: ', *lr.coef_, lr.intercept_)
def get_vif(X_design: np.ndarray) -> list:

    vif = []

    for i in range(X_design.shape[1]-1):

        vif.append(variance_inflation_factor(X_design, i+1))



    return vif
# we cook the same exp3 data

np.random.seed(42)

x1 = np.random.randn(100)

y1 = 2 * x1 + 4 # one unit change in x1 changes 2 units of y in same direction



np.random.seed(142)

x2 = np.random.randn(100)

y2 = 2 * x2 + 4 # one unit change in x2 changes 2 units of y in same direction



x3 = -x1 # one unit change in x3 changes 2 units of y in opposite direction



y = y1 + y2



X = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1)))

X.shape
X_design = sm.add_constant(X[:,:2]) # the input of ols reuires a design matrix

ols = sm.OLS(y, X_design).fit()

print(ols.summary())
print(f'\nVIF of predictors: {get_vif(X_design)}')
X_design = sm.add_constant(X)

ols = sm.OLS(y, X_design).fit()

print(ols.summary())
print(f'\nVIF of predictors: {get_vif(X_design)}')
ols = sm.regression.linear_model.OLS(y, X_design).fit_regularized(method='elastic_net', L1_wt=1.) # lasso

ols.params
ols = sm.regression.linear_model.OLS(y, X_design).fit_regularized(method='elastic_net', L1_wt=0.) # ridge

ols.params
ols = sm.regression.linear_model.OLS(y, X_design).fit_regularized(method='elastic_net', L1_wt=.5) # elastic-net

ols.params