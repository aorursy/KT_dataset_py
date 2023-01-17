#Krishna Thiyagarajan

#Abhinav Jain

#Linear Reg. Min Project

#ECE-411: Stat Learning

#Prof. Keene





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import statsmodels.api as sm

from sklearn import datasets, linear_model

from sklearn.linear_model import lasso_path

import matplotlib.pyplot as plt



np.set_printoptions(suppress = True, precision=3); #Options for NumPy



df = pd.read_csv('../input/Data.csv', sep = ',', header = 0);  #load CSV file using pandas



df.describe() # Summary of data set indicating mean, std, min etc.
corrMat = df[::].corr(); 

sns.heatmap(corrMat, vmax=0.8, square = True);
print(corrMat);


X = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data

y = df.values[:, 19]; # Last col of data

X = np.array(X); 

X = sm.add_constant(X); 

results = sm.OLS(endog=y, exog = X).fit(); 

print(results.summary())
X = df.values[:,[2,3,4,6,8,9,11,12,14,15,16,18]] # First 19 columns of data

y = df.values[:, 19]; # Last col of data

X = np.array(X); 

X = sm.add_constant(X); 

results1 = sm.OLS(endog=y, exog = X).fit(); 



print(results1.summary())
print("\n\n\nβ =", results1.params) #new beta vector for linear regression
X = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data

y = df.values[:, 19]; # Last col of data



n_alphas = 200



alphas = np.logspace(-10, -2, n_alphas)

regr = linear_model.Ridge(fit_intercept = False)



coefs = []; 

for a in alphas:

    regr.set_params(alpha = a)

    regr.fit(X,y)

    coefs.append(regr.coef_)

    

ax = plt.gca()

ax.plot(alphas, coefs)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization')

plt.axis('tight')

plt.show()
from itertools import cycle



X = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]] # First 19 columns of data

y = df.values[:, 19]; # Last col of data



eps = 5e-20

alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept = False)



plt.figure(1)

ax = plt.gca()

colors = cycle(['b', 'r', 'g', 'c', 'k']);

neg_log_alphas_lasso = -np.log10(alphas_lasso)

for coef_l, c in zip(coefs_lasso, colors):

    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c);

   

plt.xlabel('-Log(alpha)')

plt.ylabel('coefficients')

plt.title('Lasso Paths')

plt.axis('tight')









#lassocoefs = []; 

#for a in alphas:

#    regr.set_params(alpha = a)

#    regr.fit(X,y)

#    lassocoefs.append(regr.coef_)



#plt.xlabel('-Log(alpha)');

#plt.ylabel('coefficients');

#plt.title('Lasso Paths');

#plt.axis('tight');
print("Ridge   ", "Lasso    ", "OLS")



A  = [coefs[0], coefs_lasso.T[::-1][0], results.params[1:]]



print(np.matrix(A).T)