import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re



Obama_raw = pd.read_csv('../input/obama_staff_salaries.csv')

Trump_raw = pd.read_csv('../input/white_house_2017_salaries.csv')
"""Clean Trump Salary data and name columns"""



Trump = Trump_raw.copy()

Trump['SALARY'] = Trump['SALARY'].apply(lambda x: x[1:].rstrip().replace(",", ""))

Trump['SALARY'] = Trump['SALARY'].astype(float)

Trump.columns = ['Name', ' Status', 'Salary', 'Pay_basis', 'Title']





"""Clean Obama Salary data"""



Obama1 = Obama_raw.copy()

Obama2 = Obama_raw.copy()



Obama1 = Obama1.iloc[:2794]

Obama1['salary'] = Obama1['salary'].apply(lambda x: x[1:]) #cleaning salaries from 2009-2014

Obama1['salary'] = Obama1['salary'].astype(float)



Obama2 = Obama2.iloc[2794:]                                 #cleaning salaries from 2014-2016

Obama2['salary'] = Obama2['salary'].astype(float)



both_dfs = [Obama1, Obama2]

Obama = pd.concat(both_dfs)       # rejoin both dataframes

Obama.columns = ['Name', ' Status', 'Salary', 'Pay_basis', 'Title', 'Year']
Trump['Salary'].plot.hist(bins=15, color='blue', alpha=0.65)



Obama_2016 = Obama[Obama['Year'] == 2016]

Obama_2016['Salary'].plot.hist(bins=15, color='green', alpha=0.5)
Obama_2016['Salary'].plot.density()

Trump['Salary'].plot.density(color='green')
print ("Average median salaries for Trump's White House staff in 2017 so far:", "$"+str(Trump['Salary'].median()))

print ("Average median salaries for Obama's White House staff in 2016 was:   ", "$"+str(Obama_2016['Salary'].median()))
"""Compare number of Special assistants between presidencies"""



"""Trump's Executive Assistants"""

Trump[Trump['Title'].str.contains(r'\bSPECIAL ASSISTANT\b')]



"""Obama's Executive Assistants"""

Obama_2016[Obama_2016['Title'].str.contains(r'\bSPECIAL ASSISTANT\b')].head()