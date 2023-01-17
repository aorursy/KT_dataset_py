# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas as pd



df=pd.read_csv('../input/salaries-by-college-type.csv')



df.head(6)

# describe datafile details

df.describe()

#   my code

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = 10,8

import warnings

warnings.filterwarnings('ignore')

df.columns

list(df)

schools=df[['School Name', 'Starting Median Salary', 'School Type']]

schools.head(15)





NONSTR = ['Starting Median Salary']

# FIX STRING SALARY DATA TO BE NUMERIC

for x in NONSTR:

    schools[x] =schools[x].str.replace("$","")

    schools[x] =schools[x].str.replace(",","")

    schools[x] = pd.to_numeric(schools[x])

schools.head(15)    
vis = sns.barplot(data = schools, x = "School Type", y = "Starting Median Salary",)

# state type of school fares worst for starting median salary 
# enhance readability of graph with titile and legend 

vis = sns.barplot(data = schools, x = "Starting Median Salary", y = "School Type",)

vis.axes.set_title("School Starting Median Salary",fontsize=50)

vis.set_xlabel("Starting Median Salary",fontsize=30)

vis.set_ylabel("School Type",fontsize =15)

vis.tick_params(labelsize=15)
