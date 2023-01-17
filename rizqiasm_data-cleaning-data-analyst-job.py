import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head() #to get several line from top of data
data.tail() #to get several line from end of data
data = data.drop('Unnamed: 0', axis = 1)
data.head()
data = data.rename(

    columns={'Job Title': 'Job_Title',

             'Salary Estimate': 'Salary_Estimate',

             'Job Description': 'Job_Description',

             'Company Name': 'Company_Name',

             'Type of Ownership': 'Ownership',

             'Easy Apply': 'Easy_Apply'})
data.isnull().sum()
data.replace(-1, np.nan, inplace = True)

data.replace('-1', np.nan, inplace = True)

data.replace(-1.0, np.nan, inplace = True)
data.isnull().sum()
data = data.drop('Competitors', axis = 1)
data = data.drop('Easy_Apply', axis = 1)
data.head()
data = data.drop('Job_Description', axis = 1)
data.head()
data = data.Job_Title.value_counts()

data = data[0:10,]

plt.figure(figsize=(10,20))

ax = sns.barplot(data.index, data.values, alpha =0.8)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha ='right')

plt.title('Top 10 Job Title')

plt.xlabel('Job Title')

plt.ylabel('Total of Job Title')