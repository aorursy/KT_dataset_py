# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as outpu

# print(os.listdir("../input/electric-motor-temperature/pmsm_temperature_data.csv"))

%matplotlib inline  

# display the plots in same karnel
from __future__ import print_function #from print functions we are importing the future

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std
data = sm.datasets.stackloss.load(as_pandas=False)

data.exog = sm.add_constant(data.exog)

huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())

hub_results = huber_t.fit()

print(hub_results.params)

print(hub_results.bse)

print(hub_results.summary(yname='y',

            xname=['var_%d' % i for i in range(len(hub_results.params))]))



hub_results2 = huber_t.fit(cov="H2")

print(hub_results2.params)

print(hub_results2.bse)
andrew_mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.AndrewWave())

andrew_results = andrew_mod.fit(scale_est=sm.robust.scale.HuberScale(), cov="H3")

print('Parameters: ', andrew_results.params)





nsample = 50

x1 = np.linspace(0, 20, nsample)

X = np.column_stack((x1, (x1-5)**2))

X = sm.add_constant(X)

sig = 0.3   # smaller error variance makes OLS<->RLM contrast bigger

beta = [5, 0.5, -0.0]

y_true2 = np.dot(X, beta)

y2 = y_true2 + sig*1. * np.random.normal(size=nsample)

y2[[39,41,43,45,48]] -= 5   # add some outliers (10% of nsample)







res = sm.OLS(y2, X).fit()

print(res.params)

print(res.bse)

print(res.predict())



resrlm = sm.RLM(y2, X).fit()

print(resrlm.params)

print(resrlm.bse)
fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

ax.plot(x1, y2, 'o',label="data")

ax.plot(x1, y_true2, 'b-', label="True")

prstd, iv_l, iv_u = wls_prediction_std(res)

ax.plot(x1, res.fittedvalues, 'r-', label="OLS")

ax.plot(x1, iv_u, 'r--')

ax.plot(x1, iv_l, 'r--')

ax.plot(x1, resrlm.fittedvalues, 'g.-', label="RLM")

ax.legend(loc="best")





X2 = X[:,[0,1]]

res2 = sm.OLS(y2, X2).fit()

print(res2.params)

print(res2.bse)



resrlm2 = sm.RLM(y2, X2).fit()

print(resrlm2.params)

print(resrlm2.bse)
prstd, iv_l, iv_u = wls_prediction_std(res2)



fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x1, y2, 'o', label="data")

ax.plot(x1, y_true2, 'b-', label="True")

ax.plot(x1, res2.fittedvalues, 'r-', label="OLS")

ax.plot(x1, iv_u, 'r--')

ax.plot(x1, iv_l, 'r--')

ax.plot(x1, resrlm2.fittedvalues, 'g.-', label="RLM")

legend = ax.legend(loc="best")