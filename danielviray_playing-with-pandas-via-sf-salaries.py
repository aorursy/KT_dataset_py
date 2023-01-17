'''My goal for this script was to get more familiar using pandas in Python via the SF salaries data set. Also, there are some interesting statistics on SF salaries.''' 
% matplotlib inline



import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



'''Exploratory Data Analysis'''

'''Initial Inspection'''



#read in data, with 'Id' as index_col

df = pd.read_csv("../input/Salaries.csv", index_col = 'Id')



#let's see what we're working with

print ("Data types:\n\n", df.dtypes, "\n")

print ("Dimension of df:\n\n", df.shape, "\n")



#preview

df[:5]
'''Data Cleaning'''



#check how many NaN values there are in each column

df.isnull().sum()
#delete unnecessary features



#delete employee name, for sake of privacy 

del df['EmployeeName']



#delete notes, 'Notes' column in empty 

del df['Notes']



#delete 'Agency' column; all jobs are in SF

del df['Agency']



#delete any observation where JobTitle is 'Not provided'

df = df[df.JobTitle != 'Not provided']



#if 'Benefits' are NaN, fill with 0

df['Benefits'].fillna(0, inplace=True)



#Replace NaN status with "Unknown", since we don't know if its FT or PT work

df['Status'].fillna("Unknown", inplace=True)



#drop all rows with NaN values

#all missing values are in "BasePay" column

#We can afford to lose 605 observations of 148,654 obs. 

df.dropna(inplace=True)
#preview the clean data set



#no missing values

print (df.isnull().sum())



#preview data set

print (df.head(), "\n\n")

print (df.tail())
'''Exploratory Data Analysis'''



'''Top 10 Most Common Jobs in SF'''



#sum each type of observation in 'JobTitle'

job_title_counts = df['JobTitle'].value_counts()[:10]

print (job_title_counts)



#plot bar graph

job_title_counts.plot(kind = 'bar')
'''Top 10 Jobs in 2014 that pay over 300k in total'''



#filter: over 300k 

over_300k = df['TotalPayBenefits'] > 300000



#filter: year 2014

year_2014 = df['Year'] == 2014



#apply filters to dataframe, selecting only specific features

jobs_over_300k_in_2014 = df[over_300k & year_2014][['JobTitle','TotalPayBenefits', 'Year']]

top_10_jobs_over_300k_in_2014 = jobs_over_300k_in_2014[:10]



top_10_jobs_over_300k_in_2014.plot(x='JobTitle', y='TotalPayBenefits', kind='bar')
'''Distribution of top 30 jobs that pay over 300k, in 2014'''



top_30_jobs = jobs_over_300k_in_2014[['JobTitle', 'TotalPayBenefits']][:30]

dist_top_30_jobs = top_30_jobs['JobTitle'].value_counts()

dist_top_30_jobs

dist_top_30_jobs.plot(x = 'JobTitle', kind = 'bar')
'''Distribution of SF Income in 2014'''



#filter: total pay benefits in 2014

income_2014 = df[year_2014]['TotalPayBenefits']

income_2014.plot(kind='hist')
'''Percent BasePay of TotalPayBenefits'''



#create filters and store in data frame

base_pay = df[year_2014]['BasePay']

total_pay_benefits = df[year_2014]['TotalPayBenefits']



#calculate percent BasePay of TotalPayBenefits

percent_base = base_pay.astype(float) / total_pay_benefits.astype(float)



#Plot as histogram

percent_base.plot.hist(color='g', alpha = 0.5, bins = 50)
'''Aggregate Sum of SF Income'''



#create df with only Total Pay Column

df_totalpay = df['TotalPayBenefits']



#Aggregate Sum of SF Income, by year

year_counts = df.groupby('Year').aggregate(sum)

print (year_counts, "\n")



#Aggreate Sum of SF Income, by job

job_counts = df.groupby('JobTitle').aggregate(sum)

print (job_counts)