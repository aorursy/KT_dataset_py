import time

import math

import numpy as np

import pandas as pd

import scipy as sc

from scipy.optimize import minimize 

from scipy.optimize import Bounds
data = pd.read_csv("../input/dmia-sport-2019-fall-intro/train.csv", delimiter=',')
y = data['Times']

C = np.exp(np.sum(np.log(y + 1))/y.shape[0])-1
print(C)
test = pd.read_csv("../input/dmia-sport-2019-fall-intro/Xtest.csv", delimiter=',')
def save_res_const(cnst, fn):

    y_pred = np.array([cnst]*test.shape[0])

    df_res = test[['Id']]

    df_res['Times'] = y_pred

    df_res.to_csv('submission.csv', sep=',', index=None)
save_res_const(C, 'const1')