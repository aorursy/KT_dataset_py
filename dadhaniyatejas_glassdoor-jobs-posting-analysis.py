#import necessary library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#load the dataset
data = pd.read_csv('../input/glassdoor-jobs-data-analysis/glassdoor job posting test 14Oct20.csv')
data.head()
#fetch the all columns name
data.columns
#check the value counts in each columns
for i in data.columns:
    if i != 'Job Description':
        print('-->',i)
        print('-'*10)
        print(data[i].value_counts())
        print('*'*80)

#drop the Salary Estimate column and rename others since it's contain white space.
data.drop(['Salary Estimate'], axis = 1, inplace = True)

data.rename({'Job Title':'job_title', 
             'Job Description':'job_description', 
             'Rating':'rating',
             'Company Name':'company_name', 
             'Location':'location', 
             'Size':'size', 
             'Founded':'founded', 
             'Type of ownership':'type_of_ownership',
             'Industry':'industry', 
             'Sector':'sector', 
             
             
             'Revenue':'revenue'},axis = 1, inplace = True)
#company name column contain name of company and rating saperated by \n
#saperate company name and remove rating 
data['company_name'] = data['company_name'].apply(lambda x : str(x).split('\n')[0])
#number of employee working in compnay
data['size'].unique()
data['size'] = data['size'].map({'10000+ Employees':10000,
                                 '201 to 500 Employees':500,
                                 '51 to 200 Employees':200,
                                 '-1':np.nan,
                                 '5001 to 10000 Employees':10000, 
                                 '1001 to 5000 Employees':5000,
                                 '1 to 50 Employees':50,
                                 'Unknown':np.nan, 
                                 '501 to 1000 Employees':1000})
#replace -1 with NaN values
data.replace(-1,np.nan,inplace = True)
data.replace('-1',np.nan,inplace = True)
#checking null values 
data.isna().sum()
#replace Company - Public to public and Company - Private to private
data['type_of_ownership'] = data['type_of_ownership'].replace({'Company - Public':'public','Company - Private':'private'})
#replace revenue column with integer values 
data['revenue'] = data['revenue'].map({'Less than $1 million (USD)':1000000,
                                       'Unknown / Non-Applicable':np.nan,
                                       '$500 million to $1 billion (USD)':1000000000,
                                        '$10 to $25 million (USD)':25000000,
                                        '$10+ billion (USD)':10000000000,
                                        '$2 to $5 billion (USD)':5000000000,
                                        '$5 to $10 million (USD)':10000000,
                                        '$25 to $50 million (USD)':50000000,
                                        '$1 to $5 million (USD)':5000000,
                                        '$50 to $100 million (USD)':100000000,
                                        '$5 to $10 billion (USD)':10000000000,
                                        '$1 to $2 billion (USD)':2000000000,
                                        '$100 to $500 million (USD)':500000000})
#calculate hoe old company is?
data['age_of_company'] = 2020 - data['founded']
#get the information about data like type,null values, memory usage
data.info()
#top 10 job title
plt.figure(figsize=(10,7))
data['job_title'].value_counts().head(10).plot.bar()
#top 10 recruiter company 
plt.figure(figsize=(10,7))
data['company_name'].value_counts().head(10).plot.bar()
#top job location
plt.figure(figsize=(10,7))
data['location'].value_counts().plot.bar()
#types of ownerships of company
plt.figure(figsize=(10,7))
data['type_of_ownership'].value_counts().plot.bar()
#number of jobs in industry
plt.figure(figsize=(10,7))
data['industry'].value_counts().plot.bar()
#sectors portion of total jobs 
plt.figure(figsize=(10,7))
data['sector'].value_counts().plot.pie()
#visualization of how old organization 
plt.figure(figsize=(10,25))
sns.barplot(y='company_name',x='age_of_company',data=data)