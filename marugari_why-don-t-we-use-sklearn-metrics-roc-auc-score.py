import time

import numpy as np

from sklearn.metrics import roc_auc_score
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)
N = 1000000

y = np.random.choice([0, 1], N, replace=True)

x = y + 1.5 * np.random.rand(N)
time_s = time.time()

score = gini_normalized(y, x)

time_t = time.time()

print('* Gini')

print(f"score: {score}")

print(f"time:  {time_t - time_s}")
time_s = time.time()

score = roc_auc_score(y, x)

time_t = time.time()

print('* ROC-AUC')

print(f"score: {2 * score - 1}")

print(f"time:  {time_t - time_s}")