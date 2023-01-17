import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.optimize as opt

from scipy import stats

import math



data = pd.read_csv("/kaggle/input/symptomatics/data.csv",parse_dates=["Date"], dayfirst=True, index_col=0)

#plt.plot(data["Date"],data["Total"])

plt.plot(data)

plt.gcf().autofmt_xdate()
def f(x,A,k,C):

    return A*np.exp(k*x)+C



fitdata = data["2020-04-08":"2020-05-09"]

expdata = data["2020-04-08":"2020-05-21"]

(A,k,C),_ = opt.curve_fit(f,list(range(0,32)), fitdata["Total"],bounds=([900000,-1.0,100000],[1500000,-0.01,300000]))

print(A,k,C)

fit = f(np.linspace(0,43,44),A,k,C)

dffit = pd.Series(fit, index=expdata.index, name="Total")

plt.plot(dffit)

plt.plot(expdata)

plt.gcf().autofmt_xdate()
noise = expdata["Total"] - dffit

preEasingNoise = noise["2020-04-08":"2020-05-09"]

postEasingNoise = noise["2020-05-10":"2020-05-21"]

plt.plot(preEasingNoise)

plt.plot(postEasingNoise)

plt.gcf().autofmt_xdate()

print(stats.ttest_ind(preEasingNoise, postEasingNoise, equal_var = True))