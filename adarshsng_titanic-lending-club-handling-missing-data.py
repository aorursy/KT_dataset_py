import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# let's load the titanic dataset

data = pd.read_csv('../input/titanic/train.csv')



# let's inspect the first 5 rows

data.head()
# we can quantify the total number of missing values using

# the isnull method plus the sum method on the dataframe



data.isnull().sum()
# alternatively, we can use the mean method after isnull

# to visualise the percentage of

# missing values for each variable



data.isnull().mean()
# let's create a binary variable that indicates 

# whether the value of cabin is missing



data['cabin_null'] = np.where(data.Cabin.isnull(), 1, 0)
# let's evaluate the percentage of missing values in

# cabin for the people who survived vs the non-survivors.



# the variable Survived takes the value 1 if the passenger

# survived, or 0 otherwise



# group data by Survived vs Non-Survived

# and find the percentage of nulls for cabin

data.groupby(['Survived'])['cabin_null'].mean()
# another way of doing the above, with less lines

# of code :)



data['Cabin'].isnull().groupby(data['Survived']).mean()
# Let's do the same for the variable age:



# First we create a binary variable to indicates

# whether the value of Age is missing



data['age_null'] = np.where(data.Age.isnull(), 1, 0)



# and then look at the mean in the different survival groups:

data.groupby(['Survived'])['age_null'].mean()
# or the same with simpler code :)



data['Age'].isnull().groupby(data['Survived']).mean()
# In the titanic dataset, there are also missing values

# for the variable Embarked.

# Let's have a look.



# Let's slice the dataframe to show only the observations

# with missing values for Embarked



data[data.Embarked.isnull()]
# let's load the columns of interest from the

# Lending Club loan book dataset



##########################################

# Note: newer versions of pandas automatically cast strings as NA,

# so to follow along with the notebook load the data as below if using

# the latest pandas version. Loading method may need to be adjusted if

# using older versions of pandas

##########################################



data = pd.read_csv('../input/lending-club-loan-data/loan.csv',

                   usecols=['emp_title', 'emp_length'],

                   na_values='',

                   keep_default_na=False)

data.head()
# let's check the percentage of missing data

data.isnull().mean()
# let's insptect the different employer names



# number of different employers names

print('Number of different employer names: {}'.format(

    len(data.emp_title.unique())))



# a few examples of employers names

data.emp_title.unique()[0:20]
# let's inspect the variable emp_length

data.emp_length.unique()
# let's look at the percentage of borrowers within

# each label / category of emp_length variable



# value counts counts the observations per category

# if we divide by the number of observations (len(data))

# we obtain the percentages of observations per category



data.emp_length.value_counts() / len(data)
# the variable emp_length has many categories.

# I will summarise it into 3 for simplicity:

# '0-10 years' or '10+ years' or 'n/a'



# let's build a dictionary to re-map emp_length to just 3 categories:



length_dict = {k: '0-10 years' for k in data.emp_length.unique()}

length_dict['10+ years'] = '10+ years'

length_dict['n/a'] = 'n/a'



# let's look at the dictionary

length_dict
# let's re-map the emp_length variable



data['emp_length_redefined'] = data.emp_length.map(length_dict)



# let's see if it worked

data.emp_length_redefined.unique()

data.head()
# let's calculate the proportion of working years

# with the same employer for those who miss data on emp_title



# data[data.emp_title.isnull()] represents the observations

# with missing data in emp_title. I use this below:



# Calculations:

# number of borrowers for whom employer name is missing

# aka, not employed people

not_employed = len(data[data.emp_title.isnull()])



# % of borrowers for whom employer name is missing

# within each category of employment length



data[data.emp_title.isnull()].groupby(

    ['emp_length_redefined'])['emp_length'].count().sort_values() / not_employed
# let's do the same for those bororwers who reported

# the employer name



# number of borrowers for whom employer name is present:

# employed people

employed = len(data.dropna(subset=['emp_title']))



# % of borrowers within each category

data.dropna(subset=['emp_title']).groupby(

    ['emp_length_redefined'])['emp_length'].count().sort_values() / employed