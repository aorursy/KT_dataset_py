# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from sklearn.linear_model import Ridge,LinearRegression,Lasso

from sklearn.feature_selection import RFE



df = pd.read_csv('../input/FullData.csv')


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



data = df.select_dtypes(include=numerics).drop(['National_Kit','Club_Kit','Contract_Expiry'],axis = 1)



features = data.drop('Rating',axis = 1)



X = features.values



X = scale(X)

y = df[['Rating']].values.ravel()
#Use Recursive feature elimination to prune out uninformative features



lm = LinearRegression()



rfe = RFE(lm,n_features_to_select = 10)



rfe_fit = rfe.fit(X,y)



#print out the selected features

for feature in data.drop('Rating',axis = 1).columns[rfe_fit.support_]:

    print(feature)

    

X = X[:,rfe_fit.support_]

selected_features = features.loc[:, rfe_fit.support_]

n_alphas = 200

alphas = np.logspace(-4, 2, n_alphas)



clf = Lasso(fit_intercept=False)



coefs = np.zeros((X.shape[1],alphas.size))

for i,a in enumerate(alphas):

    clf.set_params(alpha=a)

    clf.fit(X, y)

    c=clf.coef_

    

    coefs[:,i] = c



    

ax = plt.gca()



ax.plot(alphas, coefs.T)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Lasso coefficients as a function of the regularization')

plt.axis('tight')

#plt.legend(selected_features.columns.tolist())

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



# Put a legend to the right of the current axis

ax.legend(selected_features.columns.tolist(),loc='center left', bbox_to_anchor=(1, 0.5))

n_alphas = 200

alphas = np.logspace(-1, 8, n_alphas)



clf = Ridge(fit_intercept=False)



coefs = np.zeros((X.shape[1],alphas.size))

for i,a in enumerate(alphas):

    clf.set_params(alpha=a)

    clf.fit(X, y)

    c=clf.coef_

    

    coefs[:,i] = c



    

ax = plt.gca()



ax.plot(alphas, coefs.T)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coefficients as a function of the regularization')

plt.axis('tight')

#plt.legend(selected_features.columns.tolist())

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])



# Put a legend to the right of the current axis

ax.legend(selected_features.columns.tolist(),loc='center left', bbox_to_anchor=(1, 0.5))
