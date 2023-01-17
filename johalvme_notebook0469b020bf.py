import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import sklearn as sk

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
data = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
data.head()
# Check the dataset dimensions

data.shape
# Describe the dataset

data.dtypes
# Review if the data has some null's values

pd.isnull(data).sum()
# Review de correlation between depent and indepent variable

data.corr()
# Shuffle the dataset

df = sk.utils.shuffle(data)
# Split dataset

train, test = train_test_split(data, test_size = 0.2)
print(len(train))

print(len(test))
lm = sm.ols(formula = 'Salary~YearsExperience', data = train).fit()
# Review the train statitstics

lm.summary()

# R-squared = 0.966 good parameter

# P-value = 

#        Intercept          1.043799e-09

#        YearsExperience    1.363448e-17

# Under thant significant level
lm.params
lm.pvalues
salary_predict = lm.predict(test['YearsExperience'])
salary_predicted = pd.DataFrame({'Salary_Predicted':salary_predict})
test_final = test.join(pd.DataFrame(salary_predicted))
test_final
lm = sm.ols(formula = 'Salary~YearsExperience', data = data).fit()
lm.summary()
s_predict = lm.predict(data['YearsExperience'])
data.join(pd.DataFrame({'Salary_Predicted':s_predict}))