# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import the necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
#shape of the dataset,
df.shape

#we have 2253 rows and 16 colums
#find the datatypes in this data set
df.dtypes
#check for duplicates. It looks like that we don't have any duplicates
df.duplicated().sum()
#we can see that there are a few columns with vlaues such as -1, -1.0, '-1'. We'll replace these with as nan for now

df=df.replace(-1,np.nan)
df=df.replace(-1.0,np.nan)
df=df.replace('-1',np.nan)
#find missing values (nan values)
df.isnull().sum()

#we can see that most of the missing data are in the columns 'Comepetitors' and 'Easy Apply'
#for easier referencing, we'll rename our columns
df.rename(columns={'Unnamed: 0':'index','Job Title': 'job_title','Salary Estimate':'salary_estimate','Job Description':'job_description',
                  'Rating':'rating','Company Name':'company_name', 'Location':'location', 'Headquarters':'headquarters','Size':'size',
                  'Founded':'founded', 'Type of ownership':'type_of_ownership', 'Industry':'industry', 'Sector':'sector','Revenue':'revenue', 'Competitors':'competitors', 'Easy Apply':'easy_apply'}, inplace=True)
#fill in the missing values
df.fillna('0', inplace=True)
df['size'].replace(0, 'Unknown')

#check to to if missing values were replaced by 'Unknown' in this case
df.isnull().sum()
#clean the company_name column which contain company name and rating and create a new column called company with just the company name
df['company'] = df['company_name'].str.replace('\n.*','')
#drop company_name and index columns
df.drop(['company_name', 'index'], axis=1, inplace=True)

# split the salary estimate column to grab the min salary
salary = df['salary_estimate'].str.split("-",expand=True,)


min_salary = salary[0]
min_salary = min_salary.str.replace('K',' ')
min_salary = min_salary.str.replace('$', ' ').fillna(0).astype('int')

df['min_salary'] = min_salary

# split the salary estimate column to grab the max salary

max_salary = salary[1]
max_salary = max_salary.str.replace('K',' ')
max_salary = max_salary.str.replace('(Glassdoor est.)',' ')
max_salary = max_salary.str.replace('$', ' ')
max_salary = max_salary.str.replace('(', ' ')
max_salary = max_salary.str.replace(')', ' ').fillna(0).astype('int')

df['max_salary'] = max_salary
#drop the salary estimate column as we don't need it anymore
df.drop('salary_estimate', axis=1, inplace=True)
#find the where min_salary is 0 and drop the row because min salary cannot be zero
df.loc[df['min_salary']==0]
df.drop(index=2149,inplace=True)
#check if it drop worked- we can see in the description that the min salary is now at 24K USD/year
df.min_salary.describe()
#clean the revenue column to get min and max revenue
df['revenue'] = df['revenue'].str.replace('$', '')
df['revenue'] = df['revenue'].str.replace(' ', '')
df['revenue'] = df['revenue'].str.replace('(USD)', '')
df['revenue'] = df['revenue'].str.replace('(', '')
df['revenue'] = df['revenue'].str.replace(')', '')
df.revenue.value_counts()
df['revenue']= df['revenue'].replace('0','Unknown/Non-Applicable' )
df['revenue'] = df['revenue'].replace('Unknown/Non-Applicable', None)

df['revenue'] = df['revenue'].str.replace('2to5billion', '2billionto5billion')
df['revenue'] = df['revenue'].str.replace('5to10billion', '5billionto10billion')
df['revenue'] = df['revenue'].str.replace('1to2billion', '1billionto2billion')
df['revenue'] = df['revenue'].str.replace('Lessthan1million', '0millionto1million')
df['revenue'] = df['revenue'].str.replace('500millionto1billion', '500millionto1billion')

df['revenue'] = df['revenue'].replace('10+billion', '10billionto11billion')


df.revenue.value_counts()
df['revenue'] = df['revenue'].str.replace('million', '')
df['revenue'] = df['revenue'].str.replace('billion', '000')


df.revenue.value_counts()
#split the revenue column to get a min and max revenue
new_revenue = df['revenue'].str.split("to",expand=True,)
min_revenue = (new_revenue[0]).astype('float64')
df['min_revenue']= min_revenue

max_revenue = (new_revenue[1]).astype('float64')
df['max_revenue']= max_revenue
df.max_revenue.fillna(0)
df.min_revenue.fillna(0)


#replace the zeroes by 'Unknown' for the type_of_ownership column
df['type_of_ownership']= df['type_of_ownership'].replace('0','Unknown')
#clean the job_title column
job_desc =df['job_title'].str.split(',', expand=True)
df['job_descrip']= job_desc[0]
df['job_descrip']=df['job_descrip'].str.replace('Sr. Data Analyst','Senior Data Analyst')
df['job_descrip']=df['job_descrip'].str.replace('Data Analyst Junior','Junior Data Analyst')
data_jobs = df.job_descrip.value_counts().head(10)
#plot a bar graph to visualize this information#find out which companies are hiring the most
top_company=df.company.value_counts().head(10)

plt.figure(figsize=(10,8))

sns.barplot(x=list(top_company.index), y=list(top_company.values), color= 'blue')
plt.title('Top 10 companies that had the highest number of hirings')
plt.xlabel('Company Name')
plt.ylabel('Number of hires')
plt.xticks(rotation = 90)
plt.show()



#salary distrubution visulization using histogram and KDE

#set matplotlib figure
f, axes = plt.subplots(1, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)


sns.distplot(df['max_salary'],  color= 'g', ax=axes[0], label = 'Maximum salary')
sns.distplot(df['min_salary'],   ax=axes[1], label = 'Minimum Salary')


plt.title('Salary disturbution for all data analyst jobs')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.xticks(rotation = 90)
plt.show()


# KDE plots for min and max salaries
sns.kdeplot(data=df['max_salary'], label='Maximum salary', shade=True)
sns.kdeplot(data=df['min_salary'], label='Minimum salary', shade=True)

# Add title
plt.title("Salary Distribution", size =16)
#find the min average salary by location
av_salary_loc_min = (df.groupby('location')['min_salary'].mean().sort_values()).head(10).astype('int')
#find the max average salary by location
av_salary_loc_max = (df.groupby('location')['max_salary'].mean().sort_values()).tail(10).astype('int')
#plot bar graph to show highest average salary by location
plt.figure(figsize =(8,6))

sns.barplot(x=list(av_salary_loc_max.index), y=list(av_salary_loc_max.values) )

plt.title('Highest average salary by location')
plt.xlabel('Name of city')
plt.ylabel('Average annual salary in USD')

plt.xticks(rotation =90)
plt.show()
#plot bar graph to show lowest average salary by location
plt.figure(figsize =(8,5))

sns.barplot(x=list(av_salary_loc_min.index),y=list(av_salary_loc_min.values))


plt.title('Lowest average salary by location')
plt.xlabel('Name of city')
plt.ylabel('Average annual salary in USD')

plt.xticks(rotation =90)
plt.show()
#calculate average revenue
average_revenue = (df.min_revenue+df.max_revenue)/2
df['average_revenue']= average_revenue
#which sectors make most profit? plot a bargraph to see
top_sector= df.groupby('sector')['average_revenue'].mean().sort_values()
top_sector=top_sector.tail(20).astype('int')

sns.barplot(x= list(top_sector.index), y= list(top_sector.values) , palette='spring')

plt.title('Top 20 sectors by annual average revenue', size =16)
plt.xlabel('Sector')
plt.ylabel('Average revenue in million USD')
plt.xticks(rotation=90)
top_industry= df.groupby('industry')['average_revenue'].mean()
top_industry=top_industry.sort_values(ascending=False).head(20).astype('int')

sns.barplot(x= list(top_industry.index), y= list(top_industry.values) , palette='autumn')

plt.title('Top 20 industries with highest average revenue', size =16)
plt.xlabel('Industry')
plt.ylabel('Average revenue in million USD')
plt.xticks(rotation=90)
#for the values 0 in the company size column, we replace them by 'Unknown'
df['size']=df['size'].replace('0', 'Unknown')
hire_comp_size= df.groupby('size')['job_title'].count()
#find the number of hires depending on the size of the company
hire_comp_size= df.groupby('size')['job_title'].count()

sns.barplot(x= hire_comp_size.index, y= hire_comp_size.values , palette='autumn')
plt.title('Number of hires by size of company', size =16)
plt.xlabel('Size if company')
plt.ylabel('Number of hires')
plt.xticks(rotation=90)

#pie chart showing the same results
explode =(0,0.1,0,0,0,0,0,0)
fig1, ax1 = plt.subplots()
ax1.pie(hire_comp_size, explode=explode, labels=hire_comp_size.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
#find number of hires by type of ownership
owner_type = (df.groupby('type_of_ownership')['job_title'].count()).sort_values().tail(5)
#pie chart showing the results
explode =(0,0.1,0,0,0,)

fig1, ax1 = plt.subplots(figsize=(100,10))
ax1.pie(owner_type, explode=explode, autopct='%1.1f%%', startangle=90 )
plt.legend(labels=owner_type.index, loc='upper right', fontsize=8)
plt.title('Percentage of hires by type of ownership of company', size=16)
plt.axis("off")

plt.show()
#number of jobs by job title
data_jobs = df.job_descrip.value_counts().head(10)

#plot bar graph to show highest average salary by location
plt.figure(figsize =(8,6))

sns.barplot(x=list(data_jobs.index), y=list(data_jobs.values) , palette='spring')

plt.title('Number of jobs by job title', size=16, fontweight='bold')
plt.xlabel('Job title', fontweight='bold')
plt.ylabel('Number of jobs',fontweight='bold')

plt.xticks(rotation =90)
plt.show()
#Word Cloud of job_titles
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

plt.subplots(figsize=(14,14))

wc = WordCloud(background_color = 'lightblue')
txt = df['job_title']
wc.generate(str(' '.join(txt)))
plt.imshow(wc)
plt.axis("off")
plt.show()