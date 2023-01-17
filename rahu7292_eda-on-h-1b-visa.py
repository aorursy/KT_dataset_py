# Importing the useful Libraries



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

%matplotlib inline

plt.rcParams['font.size'] = 15
# Reading the H-1B Dataset.



try:

    df = pd.read_csv('input/h1b_kaggle.csv')

except Exception as e :

    df = pd.read_csv('../input/h1b_kaggle.csv')

    

# Take a look at the First Five rows.



df.head()    
# Now Let's see the shape and other information about the Dataset.



print(f'The Number of Rows are-->>\t {df.shape[0]}')

print(f'The Number of Columns are-->>\t {df.shape[1]}')

print()

print(f"The CASE STATUS are:-->>\t {df['CASE_STATUS'].unique()}")

print()

print(f'The number of JOB TITLES are:-->>\t {df["JOB_TITLE"].nunique()}')

print()

print(f"The number of WORKSITES are:-->>\t {df['WORKSITE'].nunique()}")

print()

print(f"This data is from year 2011 to 2016")
# Removing the unwanted columns.



df.drop(['Unnamed: 0','lon','lat'],axis=1, inplace=True)



# Top 5 enties after removing the unwanted columns.

df.head()
# Checking whether missing data will effect our Analysis or not.



print(f'The Shape of Dataset With Missing Values is->\t {df.shape}')

print()

q = df.dropna()

print(f'The Shape of Dataset Without Missing Values is->\t {q.shape}')

print()

print("""So as One can see missing values will not effect our Analysis as we have 30 million entries.

Hence no need to remove missing values.""")
# Number of applications per year.



fig = plt.figure(figsize=(12,10), facecolor='#B0E0E6')

ax1 = fig.add_axes([0,0,.4,.4])

ax2 = fig.add_axes([.5,0,.4,.4])



year = df['YEAR'].value_counts()   # counting number of applicants in each year.

ax1.bar(x=year.index, height=year[:])

ax1.set_title('Number of H-1B Applications Year Wise')

ax1.set_xlabel('Year')

ax1.set_ylabel('Number of Applications')



#Applications growth rate per year by taking 2011 as a base year.



q = pd.Series(data=[358767 for i in range(6)])

temp_year = year[:5]

temp_year.index = [0,1,2,3,4,5]

temp_year = ((((temp_year-q)[:5])/(q[:5]))*100)  # calculating % by taking 2011 as a base year.

temp_year.index = [2016,2015,2014,2013,2012]

ax2.bar(x=temp_year.index, height=temp_year[:])

ax2.set_title("""Applicant Growth rate per year by 

taking 2011 as a base year""")

ax2.set_xlabel('Year')

ax2.set_ylabel('Growth in %')

plt.show()
# Analysing the case status and Ploting the result.



status = df['CASE_STATUS'].value_counts()



plt.figure(figsize=(10,5))

plt.pie(labels=status.index[:-3], x=status[:-3], explode=[0.2,0,0,0], autopct='%.f%%', shadow=True, textprops={'fontsize':14})

plt.show()
# Top 10 desirable job titles.



fig = plt.figure(figsize=(12,10))

ax1 = fig.add_axes([0,0,.4,.4])

ax2 = fig.add_axes([.5,0,.4,.4])

job_title = df['JOB_TITLE'].value_counts()[:10]



ax1.bar(x=job_title.index, height=job_title[:])

ax1.set_xticklabels(labels=job_title.index , rotation=90)

ax1.set_title('Top 10 Desirable Job Titles')

ax1.set_xlabel('Job Titles')

ax1.set_ylabel('Number of Applicants')



# Average Salary of top 10 desirable Job Title.



job_title_avg = []

for i in job_title.index:

    avg = df[df['JOB_TITLE']==i]['PREVAILING_WAGE'].mean()

    job_title_avg.append(avg)

    

ax2.bar(x=job_title.index, height=job_title_avg)  

ax2.set_xticklabels(labels=job_title.index, rotation=90)

ax2.set_title('Average Salary of Top 10 Job Titles')

ax2.set_xlabel('Job Titles')

ax2.set_ylabel('Average Salary')

plt.show()
# Ploting companies who sent highgest H-1B visa applications



company = df['EMPLOYER_NAME'].value_counts()[:20]

plt.figure(figsize=(15,8))

plt.bar(x=company.index, height=company[:])

plt.title('Top 20 Companies who sent highest H-1B visa applications')

plt.xlabel('Company')

plt.ylabel('Number of Applicants')

plt.xticks(rotation=90)

plt.show()
# Average salary of top 10 companies who are in demand.



company_demand = df['EMPLOYER_NAME'].value_counts()[:10]



company_demand_avg = []

for var in company_demand.index:

    avg = df[df['EMPLOYER_NAME']==var]['PREVAILING_WAGE'].mean()

    company_demand_avg.append(avg)

fig = plt.figure(figsize=(12,10), facecolor='#D3D3D3')

ax1 = fig.add_axes([0,0,.4,.4])



ax1.bar(x=company_demand.index, height=company_demand_avg[:])

ax1.set_xticklabels(labels=company_demand.index, rotation=90)

ax1.set_title("""Average salary of Top 10 companies 

who are in demand""")

ax1.set_xlabel('Company')

ax1.set_ylabel('Average Salary')



# Top 10 companies who are paying highest salary



company_highsalary = df.sort_values(by='PREVAILING_WAGE', ascending=False)['EMPLOYER_NAME'][:10]



ax2 = fig.add_axes([.5,0,.4,.4])

ax2.bar(x=company_highsalary.values, height=company_highsalary.index)



ax2.set_xticklabels(labels=company_highsalary.values, rotation=90)

ax2.set_title('Top 10 companies who Pay high salary.')

ax2.set_xlabel('Company')

ax2.set_ylabel('Salary')



plt.show()
# Top 10 desirable states by employees.



state = df['WORKSITE'].value_counts()[:10]



fig =plt.figure(figsize=(12,10), facecolor='#00FF7F')

ax1 = fig.add_axes([0,0,.4,.4])

ax1.bar(x=state.index, height=state.values)

ax1.set_xticklabels(labels=state.index, rotation=90)

ax1.set_title('TOP 10 Desirable states by employees')

ax1.set_xlabel('State')

ax1.set_ylabel('Number of applications')



# Top 5 states which have denied most?



denied_states = df[df['CASE_STATUS']=='DENIED']['WORKSITE'].value_counts()[:5]



ax2 = fig.add_axes([.5,0,.4,.4])

ax2.bar(x=denied_states.index, height=denied_states.values)

ax2.set_xticklabels(labels=denied_states.index, rotation=90)

ax2.set_title('Top 5 states ehich have denied most')

ax2.set_xlabel('State')

ax2.set_ylabel('Number of Denied Applications')



plt.show()

print()

print(f'The Correlation Between these two results is-->>\t {state.corr(denied_states)}')
# Ratio of Full Time and Half Time Position.



position = df.groupby('FULL_TIME_POSITION')['CASE_STATUS']

#print(position.count())

total = position.count()[0] + position.count()[1]

#print(total)

l = ['Half Time', 'Full Time']

l1 = [(position.count()[0]/total), (position.count()[1]/total)]

plt.figure(figsize=(10,5), )

plt.pie(labels=l, x=l1, autopct='%.f%%', explode=[0,0.2], shadow=True,textprops={'fontsize':14})

plt.show()
# Data science job profile includes job titles like Data Scientist, Data Analyst, Data Engineer, Machine Leaning Engineer,

# Business Analyst.

# Let's check out the total job created under these job titles.



jobs = ['DATA SCIENTIST', 'DATA ANALYST', 'DATA ENGINEER', 'MACHINE LEARNING ENGINEER', 'BUSINESS ANALYST']

count = []

# Counting the number of applicants related to each job title.

for var in jobs:

    q = df[df['JOB_TITLE']==var]['JOB_TITLE'].count()

    count.append(q)



job1 = ['DATA\nSCIENTIST', 'DATA\nANALYST', 'DATA\nENGINEER', 'MACHINE\nLEARNING\nENGINEER', 'BUSINESS\nANALYST']    

plt.figure(figsize=(10,5), facecolor='#B0E0E6')

plt.bar(x=job1, height=count)

plt.show()

print()

print(f"The total number of jobs related to Data Science job profile are-->>\t {sum(count)}")
# Let's calculate total jobs of all job titles related to data sciene job profile over Years.



jobs = ['DATA SCIENTIST', 'DATA ANALYST', 'DATA ENGINEER', 'MACHINE LEARNING ENGINEER', 'BUSINESS ANALYST']

total = []

# Counting number of applicants in each year for each job title.



for i in jobs:

    w = df[df['JOB_TITLE']==i].groupby('YEAR')['CASE_STATUS'].count()

    total.append(list(w.values))



total = np.array(total)



ds_total = []

# Adding all the jobs related to job profile Data Science according to year wise.



for j in range(6):

    ds_total.append(sum(total[:, j]))    

year = [2011,2012,2013,2014,2015,2016]



plt.figure(figsize=(10,5),facecolor='#FFDAB9')

plt.bar(x=year, height=ds_total)

plt.title('Growth of Data Science job profile')

plt.xlabel('Year')

plt.ylabel('Total Jobs')

plt.show()
# Average salary of job titles related to Data Science.



jobs = ['DATA SCIENTIST', 'DATA ANALYST', 'DATA ENGINEER', 'MACHINE LEARNING ENGINEER', 'BUSINESS ANALYST']

jobs_avg = []

for i in jobs:

    avg = df[df['JOB_TITLE']==i]['PREVAILING_WAGE'].mean()

    jobs_avg.append(avg)    



jobs1 = ['DATA\nSCIENTIST', 'DATA\nANALYST', 'DATA\nENGINEER', 'MACHINE\nLEARNING\nENGINEER', 'BUSINESS\nANALYST']

plt.figure(figsize=(10,5), facecolor='palegreen')

plt.bar(x=jobs1, height=jobs_avg)

plt.title('Average salary of job titles related to Data Science')

plt.xlabel('Job Titles')

plt.ylabel('Average Salary')

plt.show()

# Analysis of Top 10 company where Data Analyst applied the most.



company = df[df['JOB_TITLE']=='DATA ANALYST']['EMPLOYER_NAME'].value_counts()[:10]



plt.figure(figsize=(10,5), facecolor='#FFDAB9')

plt.bar(x=company.index, height=company.values)

plt.title('Top 10 companies where Data Analyst applied the most')

plt.xlabel('Company')

plt.ylabel('Number of Data Analyst Applicants')

plt.xticks(rotation=90)

plt.show()
# Top 10 Favourite Worksite of Data Analyst.



worksite = df[df['JOB_TITLE']=='DATA ANALYST']['WORKSITE'].value_counts()[:10]



plt.figure(figsize=(10,5), facecolor='palegreen')

plt.bar(x=worksite.index, height=worksite.values)

plt.xticks(rotation=90)

plt.title('Top 10 Favourite Worksite of Data Analyst')

plt.xlabel('Worksite')

plt.ylabel('Number of Data Analyst Applicant.')

plt.show()