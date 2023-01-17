# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
DATA = pd.read_csv('../input/HR_comma_sep.csv')

DATA.head()
#Info on column data typs

DATA.info()
#Summary statistics for the numerical fields (8/10)

DATA.describe()
#Correlation among these 8 of 10 variables

cor = DATA.corr()

sns.heatmap(cor)
#Check the effect of the categorical fields on people who have left

#Build crosstabs

pd.crosstab(DATA.sales,DATA.left).apply(lambda r: r/r.sum(), axis = 0)
#Similar analysis for salary and people leaving

pd.crosstab(DATA.left, DATA.salary)
#Check department wise salary break-up only for people who left

DATA1 = DATA[DATA.left == 1]

pd.crosstab(DATA1.sales, DATA1.salary).apply(lambda r:r/r.sum(), axis = 1)
#Converting the sales and salary to dummies and then finding the correlation

DATA2 = pd.get_dummies(DATA)

DATA2.head()
cor2 =  DATA2.corr()

sns.heatmap(cor2)
#Adding derived variables and check if it has any influence

DATA2['satisfaction_level_cube'] = DATA2['satisfaction_level']**3

cor3 = DATA2.corr()

sns.heatmap(cor3)
#Maybe across departments the reasons for leaving are different. We can explore that as well.

D1 = DATA2.groupby(['sales' , 'left'])['satisfaction_level'].mean()
DATA2 = pd.get_dummies(DATA)

cor2 = DATA2.corr()

sns.heatmap(cor2)
DATA2['satisfaction_level_cube'] = DATA2['satisfaction_level']**3

DATA2['satisfaction_level_sqrt'] = np.sqrt(DATA2['satisfaction_level'])

DATA2['satisfaction_level_log'] = np.log(DATA2['satisfaction_level'])

DATA2.hist('number_project')

#cor3 = DATA2.corr()

#sns.heatmap(cor3)
DATA1.boxplot('satisfaction_level' , by = 'salary')
#Finding the ratio of classes

DATA2['left'][DATA2.left == 1].count()/DATA2['left'].count()
#Baseline model