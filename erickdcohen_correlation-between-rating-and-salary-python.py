!pip install pyjanitor
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os

from janitor import clean_names

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
analystJobsRaw = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv").clean_names()



analystJobsRaw.head()
# Extract the salary estimate and rating from the raw df 

analystDf = analystJobsRaw[['salary_estimate', 'rating']]



# Feature engineer the average salary 

analystDf['lower_bound_salary'] = analystDf['salary_estimate'].str.extract(r'(\d{2})').astype('double') * 1000 # Make lower bound salary 

analystDf['upper_bound_salary'] = analystDf['salary_estimate'].str.extract(r'(\d{2})(?=K\s\()').astype('double') * 1000 # Make upper bound salary 

analystDf['average_salary'] = (analystDf['lower_bound_salary'] + analystDf['upper_bound_salary']) / 2



analystDf = analystDf.dropna()





analystDf.head(10)
plt.scatter(analystDf.iloc[:, 4], analystDf.iloc[:, 1])

plt.xlabel("salary")

plt.ylabel("rating")
print(analystDf.iloc[:, 4].describe())

print(analystDf.iloc[:, 1].describe())





print("Are there missing or Null data: " + str(analystDf.isnull().values.any())) 
x = analystDf.iloc[:, 4].values.reshape(-1, 1) # reshape our arrays for Linear Regression

y = analystDf.iloc[:, 1].values.reshape (-1, 1)



# Performing the Linear Regression

salaryModel = LinearRegression().fit(x, y)



# Printing the R squared correlation coefficient

print('The R squared correlation coefficient between rating and salary is: ' + str(salaryModel.score(x, y)))
x_val = sm.add_constant(analystDf.iloc[:, 1])

est = sm.OLS(analystDf.iloc[:, 4], x_val)

est2 = est.fit()

print(est2.summary())