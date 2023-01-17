# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('../input/2.01. Admittance.csv')

raw_data

data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})

data
y = data['Admitted']

x1 = data['SAT']
plt.scatter(x1,y,color='C0')

plt.xlabel('SAT',fontsize=20)

plt.ylabel('Admitted',fontsize=20)

plt.show()
x = sm.add_constant(x1)

reg_lin = sm.OLS(y,x)

results_lin = reg_lin.fit()



plt.scatter(x1,y,color ='C0')

y_hat = x1*results_lin.params[1]+results_lin.params[0]



plt.plot(x1,y_hat,lw=2.5,color='C8')

plt.xlabel('SAT', fontsize=20)

plt.ylabel('Admitted', fontsize = 20)

plt.show()
reg_log = sm.Logit(y,x)

results_log = reg_log.fit()



def f(x,b0,b1):

    return np.array(np.exp(b0+x*b1)/(1 + np.exp(b0+x*b1)))



f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))

x_sorted = np.sort(np.array(x1))



plt.scatter(x1,y,color='C0')

plt.xlabel('SAT', fontsize = 20)

plt.ylabel('Admitted', fontsize = 20)

plt.plot(x_sorted,f_sorted,color='C8')

plt.show()
x = sm.add_constant(x1)

reg_log = sm.Logit(y,x)

results_log = reg_log.fit()
results_log.summary()
x0 = np.ones(168)

reg_log = sm.Logit(y,x0)

results_log = reg_log.fit()

results_log.summary()


raw_data = pd.read_csv('../input/2.02. Binary predictors.csv')

raw_data