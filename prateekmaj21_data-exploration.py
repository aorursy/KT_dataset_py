import os

import pandas as pd

import numpy as np
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/boston-housing-dataset/Boston.csv')
data.head()
data.info()
import seaborn as sns
sns.distplot(data['CRIM'])
sns.distplot(data['ZN'])
sns.distplot(data['NOX'])
sns.distplot(data['RM'])
sns.distplot(data['AGE'])
sns.distplot(data['TAX'])
sns.distplot(data['MEDV'])
import matplotlib.pyplot as plt
plt.xlabel("Median value of owner-occupied homes in 1000 USD")

plt.ylabel("Age of the House")

plt.scatter(data['MEDV'],data['AGE'],marker=".")
plt.xlabel("Median value of owner-occupied homes in 1000 USD")

plt.ylabel("Nitric oxides concentration (parts per 10 million)")

plt.scatter(data['MEDV'],data['NOX'],marker=".")
plt.xlabel("Median value of owner-occupied homes in 1000 USD")

plt.ylabel("Full-value property-tax rate per 10,000 USD")

plt.scatter(data['MEDV'],data['TAX'],marker=".")