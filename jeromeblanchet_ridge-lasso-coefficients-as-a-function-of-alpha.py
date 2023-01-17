import time

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from scipy.stats import skew, boxcox

from scipy import stats

import statsmodels.api as sm

from pandas import DataFrame

from scipy.stats.stats import pearsonr

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def read_data(file_path):    

    print('Loading datasets...')

    X_train = pd.read_csv(file_path + 'train.csv', sep=',')

    print('Datasets loaded')

    return X_train

PATH = '../input/allstate-claims-severity/'

DATA = read_data(PATH)
DATA
all_categorical_covariates = [x for x in DATA.select_dtypes(include=['object']).columns if x not in ['id','loss', 'log_loss', 'log_loss_+_200']]

continuous_covariates = [x for x in DATA.select_dtypes(exclude=['object']).columns if x not in ['id','loss', 'log_loss', 'log_loss_+_200']]        

binary_categorical_covariates = [x for x in DATA.columns if len(DATA[x].unique()) == 2 and DATA[x].dtype == 'object']

non_bynary_categorical_covariates = [x for x in DATA.columns if len(DATA[x].unique()) > 2 and DATA[x].dtype == 'object']



print("List of Binary Categorical Covariates:\n")

print(binary_categorical_covariates)

print("\n")

print("List of Non Binary Categorical Covariates:\n")

print(non_bynary_categorical_covariates)

print("\n")

print("List of All Categorical Covariates:\n")

print(all_categorical_covariates)

print("\n")

print("List of Continuous Covariates:\n")

print(continuous_covariates)
for x in all_categorical_covariates:

    lb = LabelEncoder()

    lb.fit(DATA[x].unique())

    DATA[x] = lb.transform(DATA[x])
DATA
seed = 1



trainx = DATA.columns[1:73]

trainy = DATA.columns[-2]



X = DATA[trainx]

Y = DATA[trainy]



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)



model1 = LinearRegression(n_jobs=-1)

results1 = cross_val_score(model1, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)

print("Linear Regression (Manual Tuning): ({0:.3f}) +/- ({1:.3f})".format(-1*results1.mean(), results1.std()))
X2 = sm.add_constant(X)

model = sm.OLS(Y, X2)

model_ = model.fit()

print(model_.summary())
model2 = Ridge(alpha=1,random_state=seed)

results2 = cross_val_score(model2, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)

print("Linear Regression Ridge (Manual Tuning): ({0:.3f}) +/- ({1:.3f})".format(results2.mean(), results2.std()))
start_time = time.clock()



clf = Ridge()

coefs = []

alphas = np.logspace(-6, 9, 200)



for a in alphas:

    clf.set_params(alpha=a)

    clf.fit(X2, Y)

    coefs.append(clf.coef_)



plt.figure(figsize=(40, 20))

plt.subplot(121)

ax = plt.gca()

ax.plot(alphas, coefs, color='b')

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization Alpha parameter')

plt.axis('tight')



plt.annotate('Lasso is done at that point \nand all weights would be shrinked to zero (see proof below)', 

xy=(0.1, -0.6), xytext=(0.1, -0.4), arrowprops=dict(facecolor='black'), color='black')

plt.grid(color='black', linestyle='dotted')

plt.show()



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



clf = Ridge()

error = []

alphas = np.logspace(-6, 9, 200)



for a in alphas:

    clf.set_params(alpha=a)

    #clf.fit(X2, Y)

    error.append(cross_val_score(clf, X2, Y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1).mean())



plt.figure(figsize=(40, 20))



plt.subplot(121)

ax = plt.gca()

ax.plot(alphas, error, color='b')

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('mean absolute error')

plt.title('Mean absolute error as a function of the regularization Alpha parameter')

plt.axis('tight')



plt.annotate('Lasso is done at that point \nand all weights would be \nshrinked to zero (see proof below)', 

xy=(0.1, -0.68), xytext=(0.1, -0.65), arrowprops=dict(facecolor='black'), color='black')

plt.annotate('', 

xy=(100, -0.51), xytext=(100, -0.54), arrowprops=dict(facecolor='black'), color='black')

plt.grid(color='black', linestyle='dotted')

plt.ylim([-0.68,-0.48])

#plt.xlim([-17.5,17.5])

plt.show()



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
model3 = Lasso(alpha=0.0001,random_state=seed)

results3 = cross_val_score(model3, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)

print("Linear Regression Lasso (Manual Tuning): ({0:.3f}) +/- ({1:.3f})".format(results3.mean(), results3.std()))
start_time = time.clock()



clf = Lasso()

coefs = []

alphas = np.logspace(-6, 2, 200)



for a in alphas:

    clf.set_params(alpha=a)

    clf.fit(X2, Y)

    coefs.append(clf.coef_)



plt.figure(figsize=(40, 20))

plt.subplot(121)

ax = plt.gca()

ax.plot(alphas, coefs, color='b')

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Lasso coefficients as a function of the regularization Alpha parameter')

plt.axis('tight')



plt.annotate('', 

xy=(0.1, -0.6), xytext=(0.1, -0.4), arrowprops=dict(facecolor='black'), color='black')

plt.grid(color='black', linestyle='dotted')

plt.show()



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
start_time = time.clock()



clf = Lasso()

error = []

alphas = np.logspace(-6, 9, 200)



for a in alphas:

    clf.set_params(alpha=a)

    #clf.fit(X2, Y)

    error.append(cross_val_score(clf, X2, Y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1).mean())



plt.figure(figsize=(40, 20))



plt.subplot(121)

ax = plt.gca()

ax.plot(alphas, error, color='b')

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('mean absolute error')

plt.title('Mean absolute error as a function of the regularization Alpha parameter')

plt.axis('tight')



plt.annotate('Lasso is done at that point \nand all weights are \nshrinked to zero', 

xy=(0.1, -0.60), xytext=(0.1, -0.58), arrowprops=dict(facecolor='black'), color='black')

plt.annotate('Ridge still perform good at that point', 

xy=(100, -0.49), xytext=(100, -0.51), arrowprops=dict(facecolor='black'), color='black')

plt.grid(color='black', linestyle='dotted')

plt.ylim([-0.68,-0.48])

#plt.xlim([-17.5,17.5])

plt.show()



end_time = time.clock()

print("")

print("Total Estimation Running Time:")

print(end_time - start_time, "Seconds")
model4 = ElasticNet(alpha=0.0001,l1_ratio=0.5,random_state=seed)

results4 = cross_val_score(model4, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)

print("Linear Regression Elastic Net (Manual Tuning): ({0:.3f}) +/- ({1:.3f})".format(results4.mean(), results4.std()))
lasso = Lasso(alpha=0.0001)

lasso.fit(X_train, y_train)

ridge = Ridge(alpha=1,random_state=seed)

ridge.fit(X_train, y_train)

linear = LinearRegression()

linear.fit(X_train, y_train)

enet = ElasticNet(alpha=0.0001, l1_ratio=0.5)

enet.fit(X_train, y_train)



plt.figure(figsize = (40, 20))

plt.plot(enet.coef_, color='red', linewidth=2,label='Elastic net coefficients with α = 0.0001 & l1 Ratio = 0.5')

plt.plot(lasso.coef_, color='black', linewidth=2,label='Lasso coefficients with α = 0.0001')

plt.plot(ridge.coef_, color='green', linewidth=2,label='Ridge coefficients with α = 1')

plt.plot(linear.coef_, color='blue', linewidth=2,label='Linear coefficients with no regularization')

plt.grid(color='black', linestyle='dotted')

plt.ylim([-0.8,0.9])

plt.xlim([-5,75])

plt.legend(loc='best')

plt.title('Coefficient Distribution According to the Linear Regression type')

plt.xlabel('Binary Variable Ranked (cat1, cat2,...,cat72)')

plt.ylabel('Estimated Coefficient Value')

plt.show()