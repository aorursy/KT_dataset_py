# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# https://github.com/codebasics/py/blob/master/ML/7_logistic_reg/Exercise/7_logistic_regression_exercise.ipynb

df = pd.read_csv( '../input/HR_comma_sep.csv' )

df.head()
# Performing an analysis of the data in order to get insights that will answering the next
# question: Why the people are lefting the company ?
df.groupby( 'left' ).mean()
# Comparing the people that decided to left vs the people that still working
# the salary as main field
pd.crosstab( df.salary, df.left ).plot( kind='bar' )

# The next bar chart shows employees with high salaries are likely to not leave the company
pd.crosstab(df.Department,df.left).plot(kind='bar')

# The next chart there seem to be some impact of department on employee retention

# From the data analysis so far we can conclude that we will use following variables as dependant variables in our model
# **Satisfaction Level**
# **Average Monthly Hours**
# **Promotion Last 5 Years**
# **Salary**

subdf = df[ [ 'satisfaction_level','average_montly_hours','promotion_last_5years','salary' ] ]

subdf.head()
# Salary has all text data. It needs to be converted to numbers and we will use dummy 
# variable for that

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")

df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')

df_with_dummies.drop('salary',axis='columns',inplace=True)

df_with_dummies.head()
# Initialazing the inputs for our model

X = df_with_dummies

y = df.left
# Generation of the test and training data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
X_test.head( 20 )
predictions = model.predict( X_test.head(20) )

df_sub_input = X_test.head( 20 )

# mixing results with inputs

columns_new_name = [ "left" ]

df_results = pd.DataFrame( predictions, columns=columns_new_name )

df_final = pd.concat([ df_sub_input.reset_index(drop=True), df_results  ],axis='columns')

# In the next results you can see a relation very dependent between the people
# with low salarys and low satisfaction are lefting the company !

df_final
# Accuracy of the model

model.score(X_test,y_test)