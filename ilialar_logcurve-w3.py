# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import random



from functools import partial

from scipy.optimize import minimize

import matplotlib.pyplot as plt
def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



seed_everything(seed=0)
weights_lambda = 0.95 # reflect the weight decay for distant days

normalize = True

days_of_pace_keep = 5
print([weights_lambda ** i for i in range(100)])
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

test_df.head()
num_dates_total = len(np.unique(list(train_df['Date']) + list(test_df['Date'])))

num_dates_total

num_dates_test = len(np.unique(list(test_df['Date'])))

num_dates_test

num_dates_train = len(np.unique(list(train_df['Date'])))

num_dates_train
sample_submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

sample_submission.head()
cases = train_df['ConfirmedCases'].values.reshape((-1, num_dates_train))

fatalities = train_df['Fatalities'].values.reshape((-1, num_dates_train))
def predict(x, len_t = None):

    assert len(x) % 3 == 0

    num_cases = (len(x)) // 3

    r = np.array(x[:num_cases])

    k = np.array(x[num_cases:2 * num_cases])

    p0 = np.array(x[num_cases*2:3 * num_cases])

    t = np.arange(len_t)  

    

    exp = np.exp(t.reshape((-1,1)).dot(r.reshape((1,-1)))).transpose()



    nom = (k * p0).reshape((-1,1))*exp

    denom = p0.reshape((-1,1))*(exp - 1) + k.reshape((-1,1)) + 0.0000001

    log = nom / denom

    

    return log
def fun(x, cases = cases, regularisation = 0, lamb = weights_lambda):

    log = predict(x, cases.shape[1])

    if (log < 0).any():

        return 100000

    

    t = np.arange(cases.shape[1])

    weight = (lamb ** t)[::-1]    

    

    num_cases = (len(x)) // 3

    k = np.array(x[num_cases:2 * num_cases])

    

    result = (((np.log1p(log) - np.log1p(cases))**2)*weight).mean() / weight.mean()

    

    

    if regularisation == 0:

        return result

    else:

        r = np.array(x[:cases.shape[0]])

        mean_r = r.mean()

#         print(((r - mean_r)**2).mean())

        return result + regularisation * ((r - mean_r)**2).mean()
partial_results_x = []

partial_results_y = []



for i, case in enumerate(cases):

    x0 = [0.1] + [case[-1]*2] + [1]

    part_func = partial(fun, cases = case[None], regularisation = 0)

    bounds = []

    for j in range(len(x0)//3):

        bounds.append((0,2))

    for j in range(len(x0)//3):

        bounds.append(((case[-1]*2 - case[-days_of_pace_keep])*((case[-1]+1)/(case[-days_of_pace_keep]+1)),None)) 

    for j in range(len(x0)//3):

        bounds.append((None,None))  

    res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':100000, 'disp' : False}, bounds = bounds)

    partial_results_y.append(res.fun)

    print(i, res.fun)

    partial_results_x.append(res.x)

    

x0 = np.stack(partial_results_x).transpose().flatten()
bounds = []

for i in range(len(x0)//3):

    bounds.append((0,2))

for i in range(len(x0)//3):

    bounds.append(((cases[i,-1]*2 - cases[i,-days_of_pace_keep])*((cases[i,-1]+1)/(cases[i,-days_of_pace_keep]+1)),None)) 

for i in range(len(x0)//3):

    bounds.append((None,None))  
part_func = partial(fun, cases = cases, regularisation = 0)

res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':1000, 'disp' : True}, bounds = bounds)

print(res.fun)

x0 = res.x
part_func = partial(fun, cases = cases, regularisation = 0.3)

res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':1000, 'disp' : True}, bounds = bounds)

print(res.fun)

x0 = res.x
prediction_full = predict(x0, len_t = num_dates_total)
def draw_results(cases, prediction_full, i):

    plt.plot(cases[i])

    plt.plot(prediction_full[i])

    plt.show()



for i in range(50):

    draw_results(cases, prediction_full, i)
if normalize:

    for i in range(50):

        multiplier = (cases[:,-1] + 1).reshape(-1,1) / (prediction_full[:, num_dates_train-1] + 1).reshape(-1,1)

        prediction_full = (prediction_full +1) * multiplier - 1

        multiplier
prediction = prediction_full[:, -num_dates_test:]
for i in range(50):

    draw_results(cases, prediction_full, i)
def compute_fatalities(x, cases = cases):

    result = []

    num_cases = len(x) // 2

    lamb = np.array(x[:num_cases])

    p = np.array(x[num_cases:])

    

    fatalities = cases[:,0] * p

    result.append(fatalities)

    for j in range(1,cases.shape[1]):

        fatalities = result[-1] * lamb + cases[:,j] * p

        result.append(fatalities)

    return np.stack(result).transpose()
def fun_fat(x, fatalities = fatalities, cases = cases, regularisation = 0, lamb = weights_lambda):

    pred = compute_fatalities(x, cases)

    

    if (pred < 0).any():

        return 100000

    

    t = np.arange(fatalities.shape[1])

    weight = (lamb ** t)[::-1]    

    

    num_cases = (len(x)) // 3

    k = np.array(x[num_cases:2 * num_cases])

    

    result = (((np.log1p(pred) - np.log1p(fatalities))**2)*weight).mean() / weight.mean()

    

    

    if regularisation == 0:

        return result

    else:

        p = np.array(x[fatalities.shape[0]:])

        mean_p = p.mean()

        l = np.array(x[:fatalities.shape[0]])

        mean_l = l.mean()

#         print(((r - mean_r)**2).mean())

        return result + regularisation * (((p - mean_p)**2).mean() + ((l - mean_l)**2).mean())
partial_results_x = []

partial_results_y = []



for i, case in enumerate(cases):

    fatality = fatalities[i]

    x0 = [0.1]*2

    part_func = partial(fun_fat, cases = case[None], fatalities=fatality[None], regularisation = 0)

    bounds = []

    for j in range(len(x0)//2):

        bounds.append((0,1))

    for j in range(len(x0)//2):

        bounds.append((0,1)) 

    res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':100000, 'disp' : False}, bounds = bounds)

    partial_results_y.append(res.fun)

    print(i, res.fun)

    partial_results_x.append(res.x)
x0 = np.stack(partial_results_x).transpose().flatten()

bounds = []

for j in range(len(x0)//2):

    bounds.append((0,1))

for j in range(len(x0)//2):

    bounds.append((0,1)) 
part_func = partial(fun_fat, cases = cases, fatalities=fatalities, regularisation = 0)

res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':1000, 'disp' : True}, bounds = bounds)

print(res.fun)

x0 = res.x
for i in range(3):

    part_func = partial(fun_fat, cases = cases, fatalities=fatalities, regularisation = 0.01)

    res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':1000, 'disp' : True}, bounds = bounds)

    print(res.fun)

    x0 = res.x
for i in range(3):

    part_func = partial(fun_fat, cases = cases, fatalities=fatalities, regularisation = 0.1)

    res = minimize(part_func, x0, method='L-BFGS-B', tol=1e-6, options = {'maxiter':1000, 'disp' : True}, bounds = bounds)

    print(res.fun)

    x0 = res.x
predicted_cases = np.concatenate([cases,prediction[:, -(num_dates_total - num_dates_train):]], axis = 1)
prediction_fatalities_full = compute_fatalities(x0, predicted_cases)

prediction_fatalities_full.shape
for i in range(50):

    draw_results(fatalities, prediction_fatalities_full, i)
if normalize:

    for i in range(50):

        multiplier = (fatalities[:,-1] + 1).reshape(-1,1) / (prediction_fatalities_full[:, num_dates_train-1] + 1).reshape(-1,1)

        prediction_fatalities_full = (prediction_fatalities_full +1) * multiplier - 1
for i in range(50):

    draw_results(fatalities, prediction_fatalities_full, i)
prediction_fatalities = prediction_fatalities_full[:, -num_dates_test:]
sample_submission['ConfirmedCases'] = prediction.reshape(-1)

sample_submission['Fatalities'] = prediction_fatalities.reshape(-1)

sample_submission.to_csv('submission.csv', index = False)