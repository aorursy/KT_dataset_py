# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading the dataset 

df_14= pd.read_csv('../input/survey_2014.csv')

df_14.head()
# Look for the statiscal properties

df_14.info()
# columns for the data

df_14.columns
# Survery 2014

# no. of the Questions  - 27 questions

# how many records - 1260

# Age is the only values

df_14['Timestamp'].head()

df_14['Timestamp'].tail()

# Survey is spread around - time needs to be consdiered carefully 



# what is the age group of my data in 2014

import matplotlib.pyplot as plt 

import seaborn as sns

df_14['Age'].isna().sum() # there is no null value 

df_14['Age'].dtypes

plt.figure(figsize= (10,10))

sns.countplot(y = df_14['Age'])
# look for the age which is greater than 100

df_14[df_14['Age'] > 100]

# we have 2 outliers - 99999999999, 329 

# we need to delete these



# look for the age which is less than 0

df_14[df_14['Age']<0]



# we have 3 entires in negative, age cant be negative 

# delete these
# select a dataframe with a valid age 

df_14_Age = df_14[(df_14['Age'] < 100) & (df_14['Age'] > 0)]

df_14_Age.describe()

# plot hist plot for understanding the age distribution 

plt.figure(figsize= (10, 10))

plt.subplot(211)

df_14_Age['Age'].hist()

plt.tight_layout()

plt.subplot(212)

sns.countplot(y = df_14_Age['Age'])

# majority of the people are in thier 20 to 45
# we need to rename the columns which are questions 
# list of the new data columns which we have now 

# it wil help us to better look for the dataset

l = ['Timestamp',

'Age' ,

'Gender',

    'Country',

     'US_State', 

    'Self_Employed',

    'family_history',

    'Took_treatment',

    'work_impacted',

    'company_size',

    'remote_work',

    'company_type',

    'health_benefits_provided',

    'awareness_of_benefits',

           'wellness_program',

           'resource_health',

           'anonymity_protected',

           'medical_leave',

           'mental_negative_consequences',

           'phy_negative_consequences',

           'discuss_coworkers',

           'discuss_supervisor',

           'mental_isssue_in_interview',

           'phy_isssue_in_interview',

           'seriousness',

           'observed_negative_consequences',

           'comments']
# assign these columns to survery dataset of 2014

df_14_Age.columns = [l]

df_14_Age.head()
# load the dataset

df_16 = pd.read_csv('../input/survey_2016.csv')

df_16.head(2)
# Look for the statiscal properties

df_16.info()
# shape of the dataset 

df_16.shape



#1433 observations

# 63 questions 
# what are the int columns

df_16.info()
# age distribution for 2016

df_16['What is your age?'].describe()



# max is 323 which cant be the case. we wil consider first valid age 

df_16_Age = df_16[(df_16['What is your age?'] < 100) & (df_16['What is your age?'] > 0)]

df_16_Age['What is your age?'].hist()
# look on the all columns for 2016

df_16.columns 