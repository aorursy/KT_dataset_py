# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
from numba import jit



@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
#Remove redundant calls

def ginic(actual, pred):

    actual = np.asarray(actual) #In case, someone passes Series or list

    n = len(actual)

    a_s = actual[np.argsort(pred)]

    a_c = a_s.cumsum()

    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0

    return giniSum / n

 

def gini_normalizedc(a, p):

    if p.ndim == 2:#Required for sklearn wrapper

        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1

    return ginic(a, p) / ginic(a, a)
a = np.random.randint(0,2,600000)

p = np.random.rand(600000)

print(a[10:15], p[10:15])
gini_normalizedc(a, p)
eval_gini(a, p)
gini_normalizedc(a, p) - eval_gini(a, p)
%%timeit 

gini_normalizedc(a,p)
%%timeit 

eval_gini(a,p)
%timeit np.argsort(p)