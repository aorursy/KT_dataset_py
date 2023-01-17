import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from scipy.optimize import brute, minimize

import time
oof1 = pd.read_csv('../input/lgb-2-leaves-augment/lgb_oof.csv')

oof2 = pd.read_csv('../input/lgb-2-leaves-augment/lgb_oof.csv')



oof_tuple = tuple([oof1, oof2])
def obj_func(x, *oof_list):

    oof = 0

    x[x<0] = 0

    x = x/np.sum(x)

    for i in range(min(len(x), len(oof_list))):

        oof += x[i]*oof_list[i]['predict']

    ground_truth = oof_list[0]['target']

    return -roc_auc_score(ground_truth, oof)
obj_func(np.array([1, 1, 1, 1]), *oof_tuple)
st = time.time()

res = minimize(obj_func, x0=np.ones((2, )), args=oof_tuple, method='Nelder-Mead')

print((time.time()-st), 'sec.')
print('The optimal weights are', res.x)

print('The optimal OOF AUC is', -res.fun)