import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



print("Load the following modules; pandas, numpy")

import pandas as pd

import numpy as np



data = pd.read_csv('C:\\Users\\mascot\\Documents\\GitHub\\apple.2011.csv')

data.columns = ['Date', 'Price', 'Percent_Change'] # Renaming the columns

apple =data.copy()  #A copy of the dataframe

apple = apple[1:len(apple)]



print("Cleaning  of the data")

apple['Percent_Change'] = pd.to_numeric(apple['Percent_Change'], errors='coerce')

apple['Date'] = pd.to_datetime(apple['Date'], format='%m/%d/%Y')



apple.dtypes # data types
%%timeit

import matplotlib.pyplot as plt

plt.style.use('ggplot')



mu = apple['Percent_Change'].mean()

sigma = apple['Percent_Change'].std()

plt.hist(apple['Percent_Change'].dropna(), 20);



def gauss(n):

    return np.random.normal(mu, sigma, n)



sample = np.empty(10000)

for i in xrange(10000):

    price = apple['Price'].iloc[-1]

    percent = gauss(20)



    for day in xrange(20):

        price = price * (1 + percent[day])

        

    sample[i] = price



sample = sample.argsort()

VaR = np.percentile(sample, 1)
print(" Comparing it with Parallel. You start it with either these codes or ")

import os

import ipyparallel as ipp



rc = ipp.Client()

ar = rc[:].apply_async(os.getpid)

pid_map = ar.get_dict()


print("these codes;")

import ipyparallel as ipp



rc = ipp.Client()

rc.block = True



dview = rc.direct_view()

dview.block = False



dview.execute('import numpy as np')






%%timeit



%px import csv

%px import random

%px import numpy as np

%px sample = np.empty(10000)





apple['Percent_Change'] = pd.to_numeric(apple['Percent_Change'], errors='coerce')

apple['Date'] = pd.to_datetime(apple['Date'], format='%m/%d/%Y')



apple.dtypes



changes = []



import matplotlib.pyplot as plt

plt.style.use('ggplot')



mu = apple['Percent_Change'].mean()

sigma = apple['Percent_Change'].std()

plt.hist(apple['Percent_Change'].dropna(), 20);



def gauss(n):

    return np.random.normal(mu, sigma, n)



sample = np.empty(10000)

for i in xrange(10000):

    price = apple['Price'].iloc[-1]

    percent = gauss(20)



    for day in xrange(20):

        price = price * (1 + percent[day])

        

    sample[i] = price



sims = dview.gather('sample')

sims = np.array(sims)    

sample = sample.argsort()

VaR = np.percentile(sample, 1)




