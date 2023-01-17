# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.Series.__unicode__ = pd.Series.to_string



# reads dataset into kernel

data = pd.read_csv("../input/ntr-arogya-seva-2017/ntrarogyaseva.csv")

# displays top rows

data.head()

# primary statistics

data.describe()

# displays all columns

data.columns

# Display the counts of each value in the SEX column

data['SEX'].value_counts()

# mappings standardizes and cleans the values

mappings = {'MALE' : 'Male', 'FEMALE' : 'Female', 'Male(Child)' : 'Boy', 'Female(Child)' : 'Girl'}

# plot the value counts of sex 

data['SEX'].value_counts().plot.bar()

# print the mean, median and mode of the age distribution

print("Mean: {}".format(data['AGE'].mean()))

print("Median: {}".format(data['AGE'].median()))

print("Mode: {}".format(data['AGE'].mode()))

# print the top 10 ages

data['AGE'].value_counts().head(10)

# boxplot for age variable

#data['AGE'].plot.box()

# boxplot for age variable

data['AGE'].plot.box()
# subset involving only records of Krishna district

data[data['DISTRICT_NAME']=='Srikakulam'].head()

# Most common surgery by district

for i in data['DISTRICT_NAME'].unique():

    print("District: {}\nDisease and Count: {}".format(i,data[data['DISTRICT_NAME']==i]['SURGERY'].value_counts().head(1)))
# Average claim amount for surgery by district

for i in data['DISTRICT_NAME'].unique():

    print("District: {}\nAverage Claim Amount: ₹{}".format(i,data[data['DISTRICT_NAME']==i]['CLAIM_AMOUNT'].mean()))
# group by surgery category to get mean statistics

data.groupby('CATEGORY_NAME').mean()
# create a new memory copy of data to manipulate age 

data_age = data.copy()

# round the age variable to 0 or 1 (nearest)

data_age['AGE'] = data_age['AGE'].round(-1)

# a frequency plot for each age group

sns.countplot(data_age['AGE'])

# Most common surgery and count per age group

for i in sorted(data_age['AGE'].unique()):

    print("Age Group: {}\nMost Common Surgery and Count: {}".format(i,data[data['AGE']==i]['CATEGORY_NAME'].value_counts().head(1)))