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



# Any results you write to the current directory are saved as output.
sales= pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
sales.head()
sa= sales.groupby('date_block_num')['item_cnt_day'].apply(np.mean).reset_index()
sa.set_index('date_block_num', inplace= True)
sa.plot()

import pandas as pd

import pymc3 as pm

import matplotlib.pyplot as plt
from random import gauss

from random import seed

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# create acf plot

plot_acf(sa)

pyplot.show()
plot_pacf(sa)

pyplot.show()
!pip install arch
from random import gauss

from random import seed

from matplotlib import pyplot

from arch import arch_model
n_test = 6

train, test = sa[:-n_test], sa[-n_test:]

# define model

model = arch_model(train, mean='Zero', vol='GARCH', p=14, q=15)

# fit model

model_fit = model.fit()

# forecast the test set

yhat = model_fit.forecast(horizon=n_test)

# plot the actual variance

var = [i*0.01 for i in range(0,100)]

pyplot.plot(var[-n_test:])

# plot forecast variance

pyplot.plot(yhat.variance.values[-1, :])

pyplot.show()
n_test = 10

train, test = sa[:-n_test], sa[-n_test:]

# define model

model = arch_model(train, mean='Zero', vol='ARCH', p=14)

# fit model

model_fit = model.fit()

# forecast the test set

yhat = model_fit.forecast(horizon=n_test)

# plot the actual variance

var = [i*0.01 for i in range(0,100)]

pyplot.plot(var[-n_test:])

# plot forecast variance

pyplot.plot(yhat.variance.values[-1, :])

pyplot.show()
with pm.Model() as model:

    k_=pm.Uniform('k',-1,1)

    tau_=pm.Gamma('tau',mu=1,sd=1)

    obs=pm.AR1('observed',k=k_,tau_e=tau_,observed=sa)

    trace=pm.sample()

pm.plot_posterior(trace,'k')

print(pm.summary(trace))