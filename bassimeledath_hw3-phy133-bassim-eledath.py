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
# dependencies



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.optimize import curve_fit

plt.style.use("fivethirtyeight")
# Data



years = [2011,2012,2013,2014,2015,2016,2017,2018,2019]

kohli = [23.0,49.4,56.0,44.6,42.7,75.9,75.6,55.1,67.6]

shreyas = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,54.0,24.0,53.0]

jadeja = [np.nan,12.0,15.5,25.9,21.8,37.5,41.0,38.0,62.9]

dhoni = [27.2,40.7,45.9,33.4,np.nan,np.nan,np.nan,np.nan,np.nan]

sharma = [np.nan,np.nan,45.6,31.5,53.9,26.0,68.8,25.7,np.nan]



df = pd.DataFrame()

df['years'] = years

df['kohli'] = kohli

df['shreyas'] = shreyas

df['jadeja'] = jadeja

df['dhoni'] = dhoni

df['sharma'] = sharma
runs_std = np.std(kohli)

print(round(runs_std,3))
plt.plot(years,kohli)

plt.errorbar(years,kohli,

             yerr=runs_std)

plt.xlabel("Year")

plt.ylabel("Avg. Runs")

plt.title("Kohli's average runs over the years")

plt.show()
# 'straight line' y=f(x)



def f(x, A, B):

    return A*x + B
popt, pcov = curve_fit(f, years, kohli) # your data x, y to fit

print("Slope:",popt[0]) 

print("Intercept:",popt[1])



# y = m*x + b

y_fit = popt[0]*np.asarray(years) + popt[1]



plt.errorbar(years,kohli,

             yerr=runs_std)



# the fit!

plt.plot(years, y_fit,'--')



plt.xlabel("Year")

plt.ylabel("Avg. Runs")

plt.title("Kohli's average runs over the years")



plt.show()