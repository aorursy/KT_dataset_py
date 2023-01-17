# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



import requests

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_load = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

data_load
nevada_data = data_load[data_load.state == "Nevada"]

nevada_data = nevada_data.groupby('date', axis = 0).sum()

#washoe_county_data.drop(columns = ['date'])

nevada_data
plt.figure(figsize=(15,10))

plt.plot(nevada_data.cases)



plt.xticks(np.arange(1, 60, step=6))

plt.ylabel('Confirmed Cases', fontsize = 18)

plt.title('Nevada: Confirmed COVID-19 Cases', fontsize = 20)

plt.savefig('Nevad Cases Graph')

plt.show()



plt.figure(figsize=(15,10))

plt.plot(nevada_data.deaths)

plt.xticks(np.arange(1, 60, step=6))

plt.title('Nevada: Total Deaths From COVID-19', fontsize = 20)

plt.ylabel('Deaths', fontsize = 18)

plt.savefig('Deaths in Nevada')

plt.show()

wcd = nevada_data

y = nevada_data.cases

x = list(range(1, len(wcd)))



def fitter(x,a,b,c ,d):

    x = np.array(x)

    return d+(a*x.astype(int)+(x.astype(int)**2)*b)+((x.astype(int)**3)*c)

    

new_cases = wcd.cases - wcd.cases.shift(1)

new_cases = new_cases.dropna()

#new_cases = np.array(new_cases)



x1 = np.vectorize(x)



params, cov = curve_fit(fitter,x,new_cases)

plt.figure(figsize=(15,10))

plt.plot(x, fitter(x,*params), label = 'fitted polynomial curve')



plt.plot(new_cases, label = 'new cases')

plt.xlim(0,56)

plt.ylim(0,300)

plt.xticks(np.arange(1, 60, step=5))

plt.title('Nevada: New COVID-19 Cases by Day', fontsize = 20)

plt.ylabel('New Cases', fontsize = 18)

plt.legend(loc="lower right",fontsize=11)

plt.savefig('New Cases by Day')

plt.show()
len(nevada_data)