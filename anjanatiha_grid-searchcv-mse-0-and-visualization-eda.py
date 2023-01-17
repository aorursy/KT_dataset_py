# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn import svm

from sklearn.metrics import mean_squared_error

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/winequality-red.csv")

df.head()
df.info()
df.describe()
scaler = StandardScaler()

d = scaler.fit_transform(df) 

d = pd.DataFrame(data=d, columns=df.columns)

print(scaler.mean_)

# scaler.transform(d)
sns.pairplot(d)
# g = sns.PairGrid(d)

# g.map_diag(plt.hist)

# g.map_offdiag(plt.scatter);
g = sns.clustermap(d, 

#                    method='average',

#                    metric='euclidean', 

                   z_score=None, 

                   standard_scale=1, 

                   figsize=None, 

                   row_cluster=True, 

                   col_cluster=True, 

                   row_linkage=None, 

                   col_linkage=None, 

                   row_colors=None, 

                   col_colors=None, 

                   mask=None,

#                    cmap="mako"

                   robust=True

                  )
sns.set(style="white")

sns.clustermap(d.corr(), 

               pivot_kws=None, 

#                method='average', 

#                metric='euclidean', 

               z_score=None, 

               standard_scale=None,

               figsize=None,

               cbar_kws=None, 

               row_cluster=True, 

               col_cluster=True, 

               row_linkage=None, 

               col_linkage=None,

               row_colors=None, 

               col_colors=None, 

               mask=None,

               center=0,

               cmap="vlag",

               linewidths=.75, 

#                figsize=(13, 13)

              )

# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.set(style="white")

sns.heatmap(corr,

         vmin=None,

         vmax=None,

         cmap=cmap,

         center=None,

         robust=True,

         annot=True, 

#          fmt='.2g',

         annot_kws=None, 

#          linewidths=0.5, 

#          linecolor='white',

         cbar=True,

         cbar_kws={"shrink": .5},

         cbar_ax=None, 

         square=True, 

         xticklabels='auto',

         yticklabels='auto', 

         mask=mask, 

         ax=None)
y = df["quality"]

df.drop(["quality"], axis=1)

X = df.values
def print_performance(clf):

    # print("*"*100)

    # print("{}{}{}".format("*"*40,"Performance", "*"*40))

    print("{}".format("Performance"))

    print("*"*100)

    print("Score            : {}".format(clf.score(X, y)))

    print("Best Estimator   : {}".format(clf.best_estimator_))

    print("Best Score       : {}".format(clf.best_score_))

    print("Best Params      : {}".format(clf.best_params_))

    print("Best Index       : {}".format(clf.best_index_))

    # print("Scorer           : {}".format(clf.scorer_))

    print("Refit Time       : {}".format(clf.refit_time_))

    # print("CV Results       : {}".format(clf.cv_results_))



    params = clf.get_params()

    best_estimator = clf.best_estimator_

    cv_results = clf.cv_results_

    

    return params, best_estimator, cv_results
# parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), 

#               'degree': np.arrange(10),

#               'C':np.arrange(10)}



parameters = {'kernel':('linear', 'rbf'), 

              'degree': [1, 10],

              'C': [1, 10]}



# svr = svm.SVR(kernel='rbf',

#               degree=3, 

#               gamma='auto',

#               coef0=0.0,

#               tol=0.001,

#               C=1.0, 

#               epsilon=0.1,

#               shrinking=True,

#               cache_size=200, 

#               verbose=False, 

#               max_iter=-1)





svr = svm.SVR(gamma='auto')



clf = GridSearchCV(estimator=svr, 

                   param_grid=parameters,

                   scoring=None, 

                   fit_params=None, 

                   n_jobs=None,

                   iid='warn',

                   refit=True,

                   cv=5,

                   verbose=0,

                   pre_dispatch='2*n_jobs',

                   error_score='raise-deprecating',

                   return_train_score='warn')



clf.fit(X, y)

 

params, best_estimator, cv_results = print_performance(clf)
parameters = {'kernel':('linear', 'rbf'), 

              'degree': np.arange(1, 10),

              'C': np.arange(1, 10)}



# svr = svm.SVR(kernel='rbf',

#               degree=3, 

#               gamma='auto',

#               coef0=0.0,

#               tol=0.001,

#               C=1.0, 

#               epsilon=0.1,

#               shrinking=True,

#               cache_size=200, 

#               verbose=False, 

#               max_iter=-1)



svr = svm.SVR(gamma='auto')



clf = GridSearchCV(estimator=svr, 

                   param_grid=parameters,

                   scoring=None, 

                   fit_params=None, 

                   n_jobs=-1,

                   iid='warn',

                   refit=True,

                   cv=5,

                   verbose=1,

                   pre_dispatch='2*n_jobs',

                   error_score='raise-deprecating',

                   return_train_score='warn')





clf.fit(X, y)



params, best_estimator, cv_results = print_performance(clf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



reg = LinearRegression()



# svr = svm.SVR(kernel='linear',

#               degree=3, 

#               gamma='auto',

#               coef0=0.0,

#               tol=0.001,

#               C=1.0, 

#               epsilon=0.1,

#               shrinking=True,

#               cache_size=200, 

#               verbose=False, 

#               max_iter=-1)



# best estimator found using grid search cv

svr = svm.SVR(C=1, cache_size=200, coef0=0.0, degree=1, epsilon=0.1, gamma='auto',

  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



# clf = svr

clf = reg



print("Cross Val Score            : {}".format(cross_val_score(clf, X, y, cv=5)))



clf.fit(X_train, y_train)

print("Score (training data only) : {}".format(clf.score(X_train, y_train)))



y_pred = clf.predict(X_test)

print("Mean Squared Error         : {}".format(mean_squared_error(y_test, y_pred)))

      
x = np.arange(len(y_pred))

plt.plot(x, y_test-y_pred)

plt.title("Prediction Difference (le-14)")

plt.show()