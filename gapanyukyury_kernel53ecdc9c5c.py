import time

import math

import numpy as np

import pandas as pd

import scipy as sc

from scipy.optimize import minimize 

from scipy.optimize import Bounds
data = pd.read_csv("../input/dmia-sport-2019-fall-intro/train.csv", delimiter=',')
data_test = pd.read_csv("../input/dmia-sport-2019-fall-intro/Xtest.csv", delimiter=',')
size_test = data_test.shape[0]

size_test
data.head()
arr_a = np.array(list(map(float, data['Times'].values)))
arr_a[0:5]
arr_len = len(arr_a)

arr_len
def f_rmsle(arr_p, arr_a, arr_len):

    """

    Функция для вычисления метрики 

    """

    lst = [(math.log(p+1)-math.log(a+1))**2 for p,a in zip(arr_p, arr_a)]

    lst_sum = sum(lst)

    res = math.sqrt(lst_sum / arr_len)

    return res
def f_minimize(arr_const):

    """

    Функция для вычисления в minimize

    """

    const = arr_const[0]

    arr_p = np.full_like(arr_a, const)

    assert len(arr_p) == len(arr_a)

    return f_rmsle(arr_p, arr_a, arr_len)
init = np.array([1.3])

init
bnds = Bounds(np.array([0.5]), np.array([2.3]), keep_feasible=True)

bnds
%%time

arr_const_minimize = minimize(f_minimize, init, bounds=bnds)
arr_const_minimize
opt_const = arr_const_minimize.x[0]

opt_const
def save_res_const(cnst, fn):

    y_pred = np.array([cnst]*size_test)

    df_res = data_test[['Id']]

    df_res['Times'] = y_pred

    df_res.to_csv('data/res/' + fn + '.csv', sep=',', index=None)
save_res_const(opt_const, 'const1')