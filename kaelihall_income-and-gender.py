# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

overtime = pd.read_csv("../input/women-in-the-workplace/Employment over time.csv")

detailed= pd.read_csv("../input/women-in-the-workplace/acs-detailed-occupations-tab2.csv")

incometotal = pd.read_csv("../input/women-in-the-workplace/household-income-us-historical_csv.csv")

jobsgender = pd.read_csv("../input/women-in-the-workplace/jobsgender.csv")

education = pd.read_csv("../input/women-in-the-workplace/DetailsByEducation.csv")

jobsgender#This is the first dataset I am exploring. 


jobsgender['major_category'].value_counts()#Taking an inventory of what categories are in the Major Category column. This is too narrow for what I'm looking for. 
maincategories=jobsgender['minor_category'].value_counts()#Doing the same for minor category. This is a better sample for me.

maincategories#
mincat=jobsgender[['minor_category','total_earnings','total_workers','workers_male','total_earnings_male','workers_female','total_earnings_female']]

mincat#Creating a new variable with clean data, so that I only have  the columns necessary for my analysis 
#I start by creating a procedure which will categorize each minor category and collect the mean of all of the data in that category. 

def categorize(job):

    step1=mincat['minor_category']==job#First step is to create a boolean to determine if the command in the procedure is in value of the column for 'minor_category'

    dataset=mincat[step1]#If true then that line should be in my new dataset

    totals=dataset.mean()#I am looking to pull the mean for each column in each category

    totals.loc['minor_category']=job#I have to tell Python that the string for the 'minor_category' is just the string for the job rather than allow it to search for the mean of that string, which returns an errror. 

    return totals

    

categorize('Production')#Testing, it works!
labels=maincategories.keys()#Creating a variable to store a list of all of the minor categories

allmincats=pd.DataFrame()#Creating an empty dataframe to store my new list of means in. 

for findsums in labels:#Looping the minor categories

    newlist=categorize(findsums)#Running them through my categorize procedure

    allmincats=allmincats.append(newlist,ignore_index=True)#Entering them into my new dataframe
allmincats['percent_workers_female']=allmincats['workers_female']/allmincats['total_workers']*100#Adding percentage of the workforce column

allmincats['percent_salary_female']=allmincats['total_earnings_female']/allmincats['total_earnings']*100#Adding percentage of total earnings

allmincats#Checking it
comparews=allmincats[['minor_category','percent_workers_female','percent_salary_female']]#I am creating a dataframe which only has my two new columns in it, to explore whether percent of workforce female and percent of total salary are related

pwf=comparews.sort_values(by=['percent_workers_female'],ascending=True)#I am sorting these columns by percent of the workforce that is female, so that I will be able to identify a trend more easily through the visualization.

pwf.plot(x='minor_category',kind='bar',figsize=(12,15))#I am plotting with bars to compare side by side, and have made the y axis longer so that it will be easier to see changes in the percentages

pwf.plot(x='minor_category',kind='line',figsize=(10,15))
psf=comparews.sort_values(by=['percent_salary_female'],ascending=True)#Creating a new 

psf.plot(x='minor_category',kind='line',figsize=(15,10))
allmincats[['minor_category','total_earnings_male','total_earnings_female']].plot(kind='bar',x='minor_category',figsize=(15,7))
allmincats['salary_difs']=allmincats['total_earnings_male']-allmincats['total_earnings_female']#Creating a new column to identify the difference between men and women's salaries.

salarycomp=allmincats[['minor_category','total_earnings','salary_difs']]#Creating a new dataframe to compare total earnings with salary differences.

sc=salarycomp.sort_values(by='total_earnings',ascending=True)#Sorting the dataframe by total earnings

sc.plot(kind='bar',x='minor_category',figsize=(12,12))
sc.plot(kind='line',x='minor_category',figsize=(10,10))
education#Investigating my education dataset
education_averages=pd.DataFrame()#Creating a new dataframe in which I'd like to store just the relevant information

#The dataframe above has already calculated the averages for each minor category for me, so all I have to do is store just those values in my new dataset. 

#They are labeled with the minor acategory as their 'Occupational Category", so I am going to search for just those values.

positions=education['minor_category'].value_counts()#Creating a new series of just the minor categories

maincats=positions.keys()#Creating a list of just the keys in that series

def sorting(edu):#Creating a procedure to find the values that match the minor_category keys I stored above

    step1=education['Occupational Category']==edu#Similar pattern to my categorize procedure but I don't need to calulate mean this time.

    step2=education[step1]

    return step2 



for avgs in maincats:#Looping the categories 

    tots=sorting(avgs)#Using my procedure to these values

    education_averages=education_averages.append(tots,ignore_index=True)#Storing them in my new dataframe

    

comparelb=education_averages[['Occupational Category','mens_less_bachelors','womens_less_bachelors']]#Creating a new variable to define my dataframe with just men and women without bachelors degree data. 

comparelb['mens_less_bachelors']=comparelb['mens_less_bachelors'].str.replace(',','').astype(float)#Here's the code I used to convert the values from strings to floats

comparelb['womens_less_bachelors']=comparelb['womens_less_bachelors'].str.replace(',','').astype(float)

comparelb=comparelb.sort_values(by='womens_less_bachelors', ascending=True)#Sorting the values so patters are easier to see

comparelb.plot(kind='bar',x='Occupational Category',figsize=(10,15))

#Following the same exact pattern as the code above.

comparemb=education_averages[['Occupational Category','men_with_bachelors','women_with_bachelors']]

comparemb['men_with_bachelors']=comparemb['men_with_bachelors'].str.replace(',','').astype(float)

comparemb['women_with_bachelors']=comparemb['women_with_bachelors'].str.replace(',','').astype(float)

comparemb=comparemb.sort_values(by='women_with_bachelors', ascending=True)

comparemb.plot(kind='bar',x='Occupational Category',figsize=(10,15))

comparemb.describe()#Investigating statistics for both datasets
comparelb.describe()
compareall=education_averages[['Occupational Category','men_with_bachelors','women_with_bachelors','mens_less_bachelors','womens_less_bachelors']]#Creating a new dataframe to store all four columns in to compare to each other

compareall['men_with_bachelors']=compareall['men_with_bachelors'].str.replace(',','').astype(float)#Converting the string numbers to floats by removing their comma and casting them as floats.

compareall['women_with_bachelors']=compareall['women_with_bachelors'].str.replace(',','').astype(float)

compareall['mens_less_bachelors']=compareall['mens_less_bachelors'].str.replace(',','').astype(float)

compareall['womens_less_bachelors']=compareall['womens_less_bachelors'].str.replace(',','').astype(float)


compareall['with_bachelors_difference']=compareall['men_with_bachelors']-compareall['women_with_bachelors']

compareall['without_bachelors_difference']=compareall['mens_less_bachelors']-compareall['womens_less_bachelors']

compareall=compareall.sort_values(by='without_bachelors_difference',ascending=True)

compareall[['Occupational Category','with_bachelors_difference','without_bachelors_difference']].plot(kind='bar',x='Occupational Category',figsize=(15,10))