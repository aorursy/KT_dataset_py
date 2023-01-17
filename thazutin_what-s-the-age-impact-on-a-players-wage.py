import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import re



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

import os

print(os.listdir("../input"))



df = pd.read_csv("../input/data.csv")



df.head()
df.info()
# Fixing Financials

financial = ['Value', 'Wage', 'Release Clause']

for f in financial:

    df[f] = df[f].apply(lambda x: str(x)[1:])

    

df.head()
# K=1.000 and M=1.000.000  

def convert(value):

    regex = r'K|M'

    m = re.search(regex, value)

    if m:

        value = re.sub(regex, "", value)

        

        if m.group() == "M":

            value = pd.to_numeric(value) * 1e6

            value = value / 1000

        else:

            value = pd.to_numeric(value) * 1e3

            value = value / 1000

            

    return value

            

for f in financial:

    df[f] = df[f].apply(convert)



df.head()
df['Value'] = df['Value'].astype(int)

df['Wage'] = df['Wage'].astype(int)

df.dtypes
df.head()
numerical = df.select_dtypes(include=['float64', 'int64']).keys().values



fig = plt.figure(figsize=(25,25))

ax = fig.gca()

df[numerical].hist(ax = ax)

plt.show()
sns.jointplot(x="Age", y="Wage", data=df.sample(100), kind='reg')
run_regression(df,'Wage ~ Age')
fig = plt.figure(figsize=(4,4))

ax = fig.gca()

df['Wage'].hist(ax = ax)

plt.show()
df['Wage'].describe()
df.groupby('Age')['Wage'].size().plot()
df.groupby('Age')['Wage'].mean().plot()
#Exploring it from a log point of view



df['log_wage'] = np.log1p(df['Wage'])



df.head()
fig = plt.figure(figsize=(4,4))

ax = fig.gca()

df['log_wage'].hist(ax = ax)

plt.show()
df['log_wage'].describe()
sns.jointplot(x="Age", y="log_wage", data=df.sample(100), kind='reg')
run_regression(df,'log_wage ~ Age')
#checking the impact of dribbling



run_regression(df, 'log_wage ~ Age + Dribbling')