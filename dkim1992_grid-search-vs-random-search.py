# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns # data visualization

import matplotlib.pyplot as plt # data visualization



import scipy as sp

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')

data.sample(5)

data.info()
X = data.drop('label', axis = 1)

y = data.label



X.shape, y.shape
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.svm import SVC



def tune(X , y, search_type, n_iter):

    scores = []

    params = []

    for i in range(len(n_iter)):

        scaler = StandardScaler()

        clf = SVC()

        pipe = Pipeline(steps=[('scaler', scaler), 

                               ('svc', clf)])

        if search_type == 'grid':

            param_grid = dict(svc__C = np.logspace(-2, 5, np.round(n_iter[i]**0.5)), svc__gamma = np.logspace(-5, 1, np.round(n_iter[i]**0.5)))

            gridsearch = GridSearchCV(pipe, param_grid = param_grid, cv = 3)

            gridsearch.fit(X, y)

            scores.append(gridsearch.best_score_)

            params.append(gridsearch.best_params_)

        elif search_type == 'random':

            param_distributions = {'svc__C': sp.stats.expon(scale=10), 

            'svc__gamma': sp.stats.expon(scale=0.1)}

            randsearch = RandomizedSearchCV(pipe, param_distributions = param_distributions, n_iter= n_iter[i], cv = 3, random_state = 333)

            randsearch.fit(X, y)

            scores.append(randsearch.best_score_)

            params.append(randsearch.best_params_)

        

        print(search_type, "with", str(n_iter[i]), "iterations completed")

    

    return scores, params

n_iterations = [9, 25, 64, 100, 169]



scores_grid, params_grid = tune(X, y, 'grid', n_iterations)

scores_random, params_random = tune(X, y, 'random', n_iterations)




plt.style.use('fivethirtyeight')



plt.plot(n_iterations, scores_grid)

plt.plot(n_iterations, scores_random)



plt.legend(['Grid Search', 'Random Search'], loc='lower right')

plt.xlabel('Number of iterations')

plt.ylabel('Mean cross-validated accuracy of the best classifier')

plt.show()
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=3000,

                                   n_features=20,

                                   n_informative=10,

                                   n_redundant=4)
def tune(X , y, search_type, n_iter):

    scores = []

    params = []

    for i in range(len(n_iter)):

        if search_type == 'grid':

            param_grid = {'C' : np.logspace(-2, 5, np.round(n_iter[i]**0.5)), 'gamma' : np.logspace(-5, 1, np.round(n_iter[i]**0.5))}

            gridsearch = GridSearchCV(SVC(), param_grid = param_grid, cv = 3)

            gridsearch.fit(X, y)

            scores.append(gridsearch.best_score_)

            params.append(gridsearch.best_params_)

        elif search_type == 'random':

            param_distributions = {'C': sp.stats.expon(scale=10), 

            'gamma': sp.stats.expon(scale=0.1)}

            randsearch = RandomizedSearchCV(SVC(), param_distributions = param_distributions, n_iter= n_iter[i], cv = 3, random_state = 333)

            randsearch.fit(X, y)

            scores.append(randsearch.best_score_)

            params.append(randsearch.best_params_)

        

        print(search_type, "with", str(n_iter[i]), "iterations completed")

    

    return scores, params
n_iterations = [9, 25, 64, 100, 169]



scores_grid, params_grid = tune(X, y, 'grid', n_iterations)

scores_random, params_random = tune(X, y, 'random', n_iterations)


plt.style.use('fivethirtyeight')



plt.plot(n_iterations, scores_grid)

plt.plot(n_iterations, scores_random)



plt.legend(['Grid Search', 'Random Search'], loc='lower right')

plt.xlabel('Number of iterations')

plt.ylabel('Mean cross-validated accuracy of the best classifier')

plt.show()