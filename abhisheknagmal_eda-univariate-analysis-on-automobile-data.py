# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Importing libraries....
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/automobile-data/Automobile_data.csv')
#checking first 5 rows of dataframe.
df.head()
#checking last 5 rows of dataframe.
df.tail()
#checking no-null values and their count and type of all columns.
df.info()
#getting  summary statistics of dataframe
df.describe()
#checking which columns are of numerical type 
set(df._get_numeric_data().columns)
#checking which columns are of categorical data type
categorical_features=set(df.columns)-set(df._get_numeric_data().columns)
categorical_features
#Checking Numerical features in Univariate variables
#Histogram
# identifying a few variables of interest and checking their distribution.
x = df.horsepower
sns.distplot(x, kde=False)


# In order to make any prediction we need to fit a linear regression model, so we made sure the distribution of the variables is almost linear.
#Checking is there any skewness in distribution and outliers are present

sns.distplot(x)

# above plot looking like positively skewed
#if we remove the outliers I think it might become perfect fit for linear model
# So i will use IQR method to find outliers.

# Five Number Rule:
#  minimum (min)
#  first quartile (q1)
#  median (med) 
#  third quartile (q3)
#  maximum (max)




#median(med)
med=x.median()

#first quartile(q1)
q1=x.quantile(0.25)

#third quartile(q3)
q3=x.quantile(0.75)

# Interquartile range(iqr)= q3 -q1
iqr = q3 -q1

#maximum(max)= q3 + 1.5* iqr
max= q3 + 1.5*iqr

#minimum (min)= q1- 1.5*iqr
min=q1-1.5*iqr
#checking no of outliers
df[x>max].horsepower.count()
#It seems there are 7 outliers above the maximum
#now checking below minimum 
df[x<min].horsepower.count()
#Using IQR it shows there are no outliers are there.


#another way of checking outliers is box and whiskar plot

plt.boxplot(x)
#handling outliers
df=df[x<=max]
#Now there are no outliers
#just take a look
plt.boxplot(df.horsepower)

#Categorical features in univariate variables
#we will use COUNTPLOTS for plotting categorical features
#again importing csv because i already used df variable

plt.figure(figsize=(25,10))
sns.countplot(df['make'])

#plot shows the counts of observations in each categorical bin using bars

# If anything I have missed In EDA- Univariate analysis.
# please comment me.