import os

import pandas as pd
#Importing csv file 

d=pd.read_csv('../input/h1b_kaggle.csv')
#Doing some preliminary check on the data:
d.head()
#Dropping columns that wont be needed throughout the analysis:

d.columns
d=d.drop(['Unnamed: 0','lon','lat'],axis=1)
d[1:3]
#I will be renaming the columns to make it easier to call or perform analysis with:
d.columns=['case','employer','soc','jobTitle','fullTime','wage','year','location']
d.columns
#Checking dataframe for null values:
d.isna()
#There appears to be null values at the end of the dataframe/

#and we will be getting rid of all these rows where null is true

d=d.dropna(how='any')
d.isnull()
#Alright, now we can move on to see some summary statistics of our dataset:
d.describe()
#well something is wrong over here, let's try to fix this 
d.info()
#we need to convert year into a category dtype, which will solve what was wrong:

d.year=d['year'].astype('category')
d.info()
stat=d.describe().transpose()
stat.round(2)

#This is new section and this section will focus on Analysis and Insights:
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.show()

import warnings

warnings.filterwarnings('ignore')
#Let's start by looking at the dataframe:

d.head()
# Q. Which employers file the most petitions?

#we will be looking at the top 10 employers: 
top10=d.employer.value_counts().head(10)
#We now have a series and it goes on to show the top 10 employers that file the most petitions:

top10
top10.plot.barh(figsize=(15,5),title='Top 10 Employers Petitioning',xlim=[0,140000],xticks=np.arange(0,140000,10000))

plt.xlabel('Number of Petitions from 2011-2016')

plt.ylabel('Employers')

plt.style='dark'
# Q. What is the percentage of petitions filed by the top 10 compared to the total number of petitions?
#This calculates the total number of petitions:

d.employer.count()
#This will calculate the total number of petitions filed by the top 10 employers:

top10.sum()
#Now we can compute the statistics:

ans1=((top10.sum())/(d.employer.count()))*100


print('Percentage of petition filed by top 10 = ', ans1.round(2),'%')
# Q. What is the statistics behind the Prevailing wage of Financial Analyst over the years?
#Creating a new dataframe to make matters easier:

d1=d[['soc','wage','year']]
d1.head(4)
FinAn=d1[d1.soc=='Financial Analysts']
FinAn.count()
FinAn.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)

plt.ylim(0,250000)

#In the scatterplot above, it appears that there the years 2015-16 do not have data
#Previously going through the data, I found that there is a number of entries for financial analyst with the letter capitalized
FinAn1=d1[d1.soc=='FINANCIAL ANALYSTS']
FinAn1.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)

plt.ylim(0,250000)
#Hence we see the data for the two missing years, now we can connect both the dataframes 
frames=[FinAn,FinAn1]
FinAn2=pd.concat(frames)
FinAn2.boxplot(column='wage',by='year',figsize=(15,12),fontsize=10)

plt.ylim(0,250000)

FinAn2.head(2)
#This will compute the median wage for the years in the dataset

stat2=FinAn2.groupby('year')['wage']
stat2.median()
stat2.describe().round(2)
# Q. What positions were usually sought after by foreign workers in those years,combined? 

# for this question, I'll look into the top 10 roles/positions:

JobPos=d['jobTitle'].value_counts().head(20)
JobPos
JobPos.plot.barh(figsize=(20,10),title='Top 20 Positions Sought for Petition',xticks=np.arange(0,250000,10000),grid=True)

plt.xlabel('Number of Petitions')



plt.ylabel('Job Positions')

plt.style='dark'

plt.show()
# Q. What are the statistics for data scientists in all these years
#Since the data is large, I will be using this function to look for data scientists roles within the dataframe.

dsci=d[d.jobTitle.str.contains('DATA SCIENTIST')]
dsci.head()
dsci['case'].unique()
#I want to filter out CERTIFIED WITHDRAWN AND WITHDRAM because these were cases were either the petioner or the /

#worker opted out and I want to leave them out and only go with CERTIFIED AND DENIED.

dsci.case=dsci[(dsci['case']=='CERTIFIED')|(dsci['case']=='DENIED')]
dsci11=dsci.groupby('year')['case'].count()
dsci11.plot(figsize=(10,6),grid=True,title='Number of Petitions filled for Data Scientist Roles')

plt.xlabel('Years from 2011-2016')

plt.ylabel('Number of Petitions Filled')



plt.show()

#I'm not able to figure out why I cant insert the years in the x-axis.
# Q.Let us look at the wage statistics for data scientists
dsci.boxplot(column='wage',by='year',figsize=(20,15),fontsize=15)

plt.ylim(0,250000)

plt.xlabel('Year')

plt.ylabel('Wage')
dsci.head(2)
statDS=dsci.groupby('year')['wage']
statDS.median()
statDS.describe()
statDS.mean().round(2)