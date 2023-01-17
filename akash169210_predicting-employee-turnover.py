# Randy Lao 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# import necessacry libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/HR_comma_sep.csv")
df.describe(include="all")
df.isnull().any()
# Get aquick overview of what we are dealig with our dataset 

df.head()
# Rename columns for better readability

df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                       'time_spend_company': 'yearsAtCompany',
                       'Work_accident': 'workAccident', 
                        'promotion_last_5years': 'promotion',
                        'sales': 'department',
                        'left': 'turnover',
                        })
# Move the reponse variable "turnover" to the front of the table
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
df.head()
#3 Statistical Overview
df.shape
# check the type of our features
df.dtypes
#Looks like about 76% of employees stayed and 24% of employees left
# Note: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = df.turnover.value_counts() / len(df)
turnover_rate
# Display the statistical overview of the employees 
df.describe()
#Overview of summary (turnover V.S. Non-turnover)
turnover_Summary = df.groupby('turnover')
turnover_Summary.mean()
#Correlation Matrix 
corr = df.corr()
corr = (corr)
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)

corr
# Let's compare the means of our employee turnover satisfaction against the employee population ssatisfaction
#emp_population = df['satisfaction'].mean()
emp_population =df['satisfaction'][df['turnover'] ==0].mean()
emp_turnover_satisfaction = df[df['turnover']==1]['satisfaction'].mean()

print( 'The mean satisfaction for the employee population with no turnover is: '+ str(emp_population))
print( 'The mean satisfaction for employees that had a turnover is: '+ str(emp_turnover_satisfaction) )
