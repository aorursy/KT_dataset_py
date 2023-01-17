# import library

import pandas as pd
# load data

df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
# let's check the data

df.head()
# What is the shape of data?

df.shape
# check for the null values

df.isnull().sum()
# check out the unique values of gender

df['gender'].unique()
# total number of values in gender

df['gender'].value_counts()
# check out the unique values of race

df['race/ethnicity'].unique()
# total number of values in race

df['race/ethnicity'].value_counts()
# check out the unique values

df['parental level of education'].unique()
# total number of values in parental education

df['parental level of education'].value_counts()
# check out the unique values of lunch

df['lunch'].unique()
# total number of values in lunch

df['lunch'].value_counts()
# check out the unique values

df['test preparation course'].unique()
# total number of values 

df['test preparation course'].value_counts()
df.groupby(['race/ethnicity']).sum()