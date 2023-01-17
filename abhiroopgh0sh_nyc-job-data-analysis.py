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
import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import os

%matplotlib inline

import numpy as np
print(os.listdir('../input'))
ds = pd.read_csv("../input/nyc-jobs.csv")
ds.head()
## Count jobs based on Posting Type

ds['Posting Type'].value_counts()
ds['Posting Type'].value_counts().plot(kind='bar')

plt.title("Total job count based on job position")

plt.ylabel('Job count')

plt.xlabel('Job position')

plt.show()
## Conclusion : Internal job vacancies are more than External jobs.
label=['min_salary','max_salary']

def salary_based_on_job_category(job,salary_freq):

     sal_freq_data=ds[ds['Salary Frequency']==salary_freq]

     job_cat_data=sal_freq_data[sal_freq_data['Job Category']==job]

     min_salary=job_cat_data['Salary Range From']

     max_salary=job_cat_data['Salary Range To']

     avg_max_salary=sum(max_salary)/len(max_salary)

     print("Count of "+salary_freq+" Job position in ("+job+") is :",len(max_salary))

     avg_min_salary=sum(min_salary)/len(min_salary)

     plt.bar(label,[avg_min_salary,avg_max_salary])

     plt.title("Average Min Max "+salary_freq+" Salary for "+job)

     plt.show()

     print("Minimum Avg salary for the category:",avg_min_salary)

     print("Maximum Avg salary for the category:",avg_max_salary)
salary_based_on_job_category('Engineering, Architecture, & Planning','Annual')
salary_based_on_job_category('Engineering, Architecture, & Planning','Hourly')
agency = ds['Agency'].value_counts()

#top 5 agencies providing highest job vacancies

top_agency = agency[:5]

top_agency.plot(kind='barh',alpha=0.6, figsize=(12,15))

plt.show()
#top 5 job titles with highest job vacancies

job_titles = ds['Civil Service Title'].value_counts()

top_job_titles = job_titles[:5]

top_job_titles.plot(kind='barh',alpha=0.6, figsize=(12,15))

plt.show()
ds.dtypes
# Converting posting dat object to datetime

ds[["Posting Date"]] = ds[["Posting Date"]].apply(pd.to_datetime)
# Renaming columns to remove spaces in the coulmn names

ds.rename(columns={'Job ID': 'job_no','Full-Time/Part-Time indicator':'shift_indicator'}, inplace=True)
#Now displaying the updated datatype

ds.dtypes
#Yearwise job posting

var = ds.groupby('Posting Date').job_no.sum()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Posting Date')

ax1.set_ylabel('Job posted')

ax1.set_title("Yearwise wise job posted")

var.plot(kind='line')
#Change of salary frequency on shift changes F is fulltime P is parttime

var = ds.groupby(['Salary Frequency','shift_indicator']).job_no.sum()

var.unstack().plot(kind='bar',stacked=True,  color=['green','orange'], alpha=0.6, figsize=(12,15))