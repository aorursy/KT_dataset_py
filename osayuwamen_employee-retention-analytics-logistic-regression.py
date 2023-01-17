# IMPORT THE RELEVANT LIBRARIES
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# LOAD THE DATA
raw_data = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

raw_data
raw_data.columns
# DECLARE THE DEPENDENT AND INDEPENDENT VARIABLES 
x1 = raw_data[['satisfaction_level', 'average_montly_hours','promotion_last_5years','salary']]

y = raw_data['left']
# FIND MISSING VALUES AND EXPLORE THE DATA TOKNOW THE RELEVANT FEATURES
raw_data.isnull().sum()
pd.crosstab( raw_data.salary, raw_data.left ).plot( kind='bar' )
pd.crosstab( raw_data.Department, raw_data.left ).plot( kind='bar' )
pd.crosstab( raw_data.promotion_last_5years, raw_data.left ).plot( kind='bar' )
pd.crosstab( raw_data.average_montly_hours, raw_data.left ).plot( kind='bar' )
# CONVERT SALARY FROM TEXT TO NUMBERS WITH DUMMIES
x_salary_dummies = pd.get_dummies(x1['salary'])
x_with_dummies = pd.concat([x1,x_salary_dummies], axis =1)
x = x_with_dummies.drop('salary', axis=1)

y = raw_data['left']

x
# SPLIT INTO TRAIN AND TEST DATA 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)
# CREATE THE MODEL
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()

reg_log.fit(x,y)
# CHECK THE ACCURACY  OF THE MODEL
reg_log.score(x,y)