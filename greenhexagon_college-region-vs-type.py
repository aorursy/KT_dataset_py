# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



college_type = pd.read_csv('../input/salaries-by-college-type.csv')

college_region = pd.read_csv('../input/salaries-by-region.csv')
college_type.head(5)
college_region.head(5)
#Combine the college data

cols = ['School Name','Region']

college_combined = pd.merge(left=college_type, right=college_region[cols], how='inner',on='School Name')

college_combined.head()
dollar_cols = ['Starting Median Salary','Mid-Career Median Salary']



for x in dollar_cols:

    college_combined[x] = college_combined[x].str.replace("$","")

    college_combined[x] = college_combined[x].str.replace(",","")

    college_combined[x] = pd.to_numeric(college_combined[x])

pivotinfo = pd.pivot_table(college_combined,index=['Region'],columns=['School Type'], values =['Starting Median Salary'])

sns.heatmap(pivotinfo, annot=True)
pivotinfo = pd.pivot_table(college_combined,index=['Region'],columns=['School Type'], values =['Mid-Career Median Salary'])

sns.heatmap(pivotinfo, annot=True)