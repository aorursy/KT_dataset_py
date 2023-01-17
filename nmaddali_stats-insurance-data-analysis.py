# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#1 Step #1. Import the necessary libraries
import pandas as pd

import numpy as np

import seaborn as sns

import os

import matplotlib.pyplot as plt

import scipy.stats as stats
#2 Step #2. Read the data as a data frame
inc_exp = pd.read_csv("/kaggle/input/insurance.csv")
inc_exp.head(10)
#3 Step #3.a Shape of data
inc_exp.shape
inc_exp.size
#4 Step #3.b Data type of each attribute
inc_exp.dtypes
inc_exp.info()
#5 Step #3.c Checking the presence of missing values
#Check NULLs in full dataset as summary

pd.isnull(inc_exp).sum()
#Check NULLs in 1st 10 records

pd.isnull(inc_exp.head(10))
#Check NULLs in full dataset - Detailed

pd.isnull(inc_exp)
#6 Step #3.d 5-point summary of numerical attributes
inc_exp.describe()
#7 Step #3.e Distribution of ‘bmi’, ‘age’ and ‘charges’ columns.
#Distribution of ‘bmi’

num_bins = inc_exp['bmi'].count()

plt.hist(inc_exp['bmi'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
#Distribution of ‘age’

num_bins = inc_exp['age'].count()

plt.hist(inc_exp['age'], num_bins, density=1, facecolor='green', alpha=0.5)

plt.show()
#Distribution of ‘charges’

num_bins = inc_exp['charges'].count()

plt.hist(inc_exp['charges'], num_bins, density=1, facecolor='red', alpha=0.5)

plt.show()
fig = plt.figure(figsize=(10,6))



ax1 = fig.add_subplot(5,1,1) 

inc_exp[['bmi']].plot(kind="density", figsize=(5,5), ax=ax1)

plt.vlines(inc_exp[['bmi']].mean(), ymin=0, ymax=0.07, linewidth=2.0, color='black')

plt.vlines(inc_exp[['bmi']].median(), ymin=0, ymax=0.07, linewidth=2.0, color='red')



ax2 = fig.add_subplot(5,1,3) 

inc_exp[['age']].plot(kind="density", figsize=(5,5), ax=ax2)

plt.vlines(inc_exp[['age']].mean(), ymin=0, ymax=0.05, linewidth=2.0, color='black')

plt.vlines(inc_exp[['age']].median(), ymin=0, ymax=0.05, linewidth=2.0, color='red')



ax3 = fig.add_subplot(5,1,5) 

inc_exp[['charges']].plot(kind="density", figsize=(5,5), ax=ax3)

plt.vlines(inc_exp[['charges']].mean(), ymin=0, ymax=0.00007, linewidth=2.0, color='black')

plt.vlines(inc_exp[['charges']].median(), ymin=0, ymax=0.00007, linewidth=2.0, color='red')
#8 Step #3.f Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns
inc_exp["bmi"].skew(), inc_exp["age"].skew(), inc_exp["charges"].skew()
inc_exp["bmi"].kurt(), inc_exp["age"].kurt(), inc_exp["charges"].kurt()
#9 Step #3.g Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns
#The presence of outliers in ‘bmi’

sns.boxplot(x='children',y='bmi',data=inc_exp)
#The presence of outliers in ‘age’

sns.boxplot(x='children',y='age',data=inc_exp)
#The presence of outliers in ‘charges’

sns.boxplot(x='children',y='charges',data=inc_exp)
#10 Step #3.h Distribution of categorical columns (include children)
inc_exp.describe(include='all')
inc_exp.info()
# inc_cat = inc_exp.select_dtypes(include = 'object').copy() #only Categorical Cols

inc_cat = inc_exp[['sex', 'children', 'smoker', 'region']]

inc_cat.head(3)
fig = plt.figure(figsize=(10,6))



ax1 = fig.add_subplot(5,1,1) 

sns.countplot(data = inc_cat, x = 'sex', ax=ax1)



ax2 = fig.add_subplot(5,1,3) 

sns.countplot(data = inc_cat, x = 'smoker', ax=ax2)



ax3 = fig.add_subplot(5,1,5) 

sns.countplot(data = inc_cat, x = 'region', ax=ax3)
fig = plt.figure(figsize=(10,6))



ax1 = fig.add_subplot(5,1,1) 

sns.boxplot(x='sex',y='children',data=inc_cat, ax=ax1)



ax2 = fig.add_subplot(5,1,3) 

sns.boxplot(x='smoker',y='children',data=inc_cat, ax=ax2)



ax3 = fig.add_subplot(5,1,5) 

sns.boxplot(x='region',y='children',data=inc_cat, ax=ax3)
#11 Step #3.i Pair plot that includes all the columns of the data frame
sns.pairplot(inc_exp)
#12 Step #4.a Do charges of people who smoke differ significantly from the people who don't?
inc_exp[inc_exp.smoker == 'yes'][['charges']].mean(), inc_exp[inc_exp.smoker == 'no'][['charges']].mean()
inc_exp[inc_exp.smoker == 'yes'][['charges']].median(), inc_exp[inc_exp.smoker == 'no'][['charges']].median()
plt.figure(figsize=(10,6))

sns.swarmplot(x='smoker',y='charges',data=inc_exp[['smoker', 'charges']])

plt.show()
# YES

#Charges of people who smoke differ significantly from the people who don't.
#13 Step #4.b Does bmi of males differ significantly from that of females?
inc_exp[inc_exp.sex == 'male'][['bmi']].mean(), inc_exp[inc_exp.sex == 'female'][['bmi']].mean()
inc_exp[inc_exp.sex == 'male'][['bmi']].median(), inc_exp[inc_exp.sex == 'female'][['bmi']].median()
plt.figure(figsize=(10,6))

sns.swarmplot(x='sex',y='bmi',data=inc_exp[['sex', 'bmi']])

plt.show()
# NO 

#bmi of males DOESN'T differ significantly from that of females
#14 Step #4.c Is the proportion of smokers significantly different in different genders?
inc_prop = inc_exp[inc_exp.smoker == 'yes']

inc_prop1 = inc_prop[['smoker','sex']]

inc_prop_f, inc_prop_m = inc_prop1[inc_prop1.sex == 'female'], inc_prop1[inc_prop1.sex == 'male']

prop_m = inc_prop_m["sex"].count()/inc_exp[inc_exp.smoker == 'yes']["smoker"].count()

prop_f = inc_prop_f["sex"].count()/inc_exp[inc_exp.smoker == 'yes']["smoker"].count()

prop_m, prop_f
# NO

#the proportion of smokers IS NOT significantly different in different genders?
#15 Step #4.d Is the distribution of bmi across women with no children, one child and two children, the same ?
inc_exp_f = inc_exp[inc_exp.sex == 'female']

fig = plt.figure(figsize=(10,6))



ax1 = fig.add_subplot(5,1,1)

inc_ch0 = inc_exp_f[inc_exp_f.children == 0]

inc_ch0[['bmi']].plot(kind="density", figsize=(5,5), ax=ax1)

plt.vlines(inc_ch0[['bmi']].mean(), ymin=0, ymax=0.07, linewidth=2.0, color='black')

plt.vlines(inc_ch0[['bmi']].median(), ymin=0, ymax=0.07, linewidth=2.0, color='red')



ax2 = fig.add_subplot(5,1,3) 

inc_ch1 = inc_exp_f[inc_exp_f.children == 1]

inc_ch1[['bmi']].plot(kind="density", figsize=(5,5), ax=ax2)

plt.vlines(inc_ch1[['bmi']].mean(), ymin=0, ymax=0.07, linewidth=2.0, color='black')

plt.vlines(inc_ch1[['bmi']].median(), ymin=0, ymax=0.07, linewidth=2.0, color='red')



ax3 = fig.add_subplot(5,1,5) 

inc_ch2 = inc_exp_f[inc_exp_f.children == 2]

inc_ch2[['bmi']].plot(kind="density", figsize=(5,5), ax=ax3)

plt.vlines(inc_ch2[['bmi']].mean(), ymin=0, ymax=0.07, linewidth=2.0, color='black')

plt.vlines(inc_ch2[['bmi']].median(), ymin=0, ymax=0.07, linewidth=2.0, color='red')
# The distribution of bmi across women with no children, one child and two children are VERY MUCH SIMILAR.