# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from bayes_opt import BayesianOptimization

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data, target = make_classification(n_samples=2500,

                                   n_features=45,

                                   n_informative=15,

                                   n_redundant=5)

def sample_loss(p_c,p_gamma):

  C = p_c

  gamma = p_gamma



  # Sample C and gamma on the log-uniform scale

  model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)



  # Sample parameters on a log scale

  return cross_val_score(model,

                         X=data,

                         y=target,

                         scoring='roc_auc',

                         cv=3).mean()

lambdas = np.linspace(1, -4, 25)

gammas = np.linspace(1, -4, 20)



# We need the cartesian combination of these two vectors

param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])

print("start grid search")

start = time.clock()

real_loss = [sample_loss(params[0],params[1]) for params in param_grid]

end = time.clock()



# The maximum is at:

param_grid[np.array(real_loss).argmax(), :]
max(real_loss)
print("the executed time is {}".format(end-start))
optimizer = BayesianOptimization(

    f = sample_loss,

    pbounds = {'p_c':(-4,1),'p_gamma':(-4,1)},

    random_state=1234,

    verbose=2

)

print("Executed time of bayesian optimization")

bo_start = time.clock()

optimizer.maximize(n_iter = 10)

bo_end = time.clock()
para = optimizer.max['params']
optimizer.max
print(bo_end-bo_start)
C = para['p_c']

gamma = para['p_gamma']



# Sample C and gamma on the log-uniform scale

BO_model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)



# Sample parameters on a log scale

a =  cross_val_score(BO_model,

                     X=data,

                     y=target,

                     scoring='roc_auc',

                     cv=3).mean()

print(a)