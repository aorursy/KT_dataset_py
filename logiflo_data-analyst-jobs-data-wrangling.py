# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv', index_col=0)
data.head()
data.info()
data['Salary Estimate'].unique()
data['Salary Estimate'] = data['Salary Estimate'].replace('-1', None)
data['Salary Estimate'] = data['Salary Estimate'].apply(lambda x:x.split(' ')[0])
data['Salary Estimate'] = data['Salary Estimate'].str.replace('$', '')
data['Salary Estimate'] = data['Salary Estimate'].str.replace('K', '000')
data['Salary Estimate'].unique()
data['Min Salary'] = data['Salary Estimate'].apply(lambda x:x.split('-')[0])
data['Max Salary'] = data['Salary Estimate'].apply(lambda x:x.split('-')[1])

data['Min Salary'] = data['Min Salary'].astype('float64')
data['Max Salary'] = data['Max Salary'].astype('float64')
data['Average Salary'] = data[['Min Salary', 'Max Salary']].mean(axis = 1)
data.head()
data['Company Name'] = data['Company Name'].astype('str')
data['Company Name'] = data['Company Name'].apply(lambda x:x.split('\n')[0])

data['Location'] = data['Location'].apply(lambda x:x.split(',')[1])
data.groupby(['Location']).mean()
data['Size'].unique()
data['Size'] = data['Size'].replace('Unknown', None)
data['Size'] = data['Size'].replace('-1', None)
data['Size'] = data['Size'].replace('10000+ employees', '10000 to 10000 employees')
data['Min Size'] = data['Size'].apply(lambda x:x.split(' to ')[0])
data['Max Size'] = data['Size'].apply(lambda x:x.split(' to ')[1])
data['Max Size'] = data['Max Size'].apply(lambda x:x.split(' ')[0])
data['Min Size'] = data['Min Size'].astype('float64')
data['Max Size'] = data['Max Size'].astype('float64')

data['Revenue'] = data['Revenue'].replace('Unknown / Non-Applicable', None)
data['Revenue'] = data['Revenue'].replace('-1', None)
data['Revenue'] = data['Revenue'].str.replace('$','')
data['Revenue'] = data['Revenue'].replace('Less than 1 million (USD)', '0 to 1 million (USD)')
data['Revenue'] = data['Revenue'].replace('10+ billion (USD)', '10000 to 10000 million (USD)')
data['Revenue'] = data['Revenue'].replace('500 million to 1 billion (USD)', '500 to 1000 million (USD)')
data['Revenue'] = data['Revenue'].replace('2 to 5 billion (USD)', '2000 to 5000 million (USD)')
data['Revenue'] = data['Revenue'].replace('5 to 10 billion (USD)', '5000 to 10000 million (USD)')
data['Revenue'] = data['Revenue'].replace('1 to 2 billion (USD)', '1000 to 2000 million (USD)')


data['Revenue'].unique()
data['Min Revenue'] = data['Revenue'].apply(lambda x:x.split(' to ')[0])
data['Max Revenue'] = data['Revenue'].apply(lambda x:x.split(' to ')[1])
data['Max Revenue'] = data['Max Revenue'].apply(lambda x:x.split(' ')[0])
data['Min Revenue'] = data['Min Revenue'].astype('float64')
data['Max Revenue'] = data['Max Revenue'].astype('float64')
data.head()
data['Num Competitors'] = data['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)

data['Easy Apply'] = data['Easy Apply'].replace('True', 'Yes')
data['Easy Apply'] = data['Easy Apply'].replace('-1', 'No')
data['Easy Apply'].unique()
data[['Num Competitors', 'Competitors']]
data['Job Title'] = data['Job Title'].str.replace('Sr.', 'Senior')
data['Job Title'] = data['Job Title'].str.replace('Jr.', 'Junior')
data['Job Title'] = data['Job Title'].apply(lambda x:x.split(',')[0])
data['Job Title']


from collections import Counter
jobs = Counter(data['Job Title'])

most_common_jobs=jobs.most_common(8)
x,y= zip(*most_common_jobs)
x,y= list(x), list(y)

plt.figure(figsize=(15,10))
sns.barplot(x=x,y=y)
plt.xlabel('Most Common Job Titles')
plt.figure(figsize=(10,10))
plt.pie(y, labels = x, autopct='%1.1f%%', labeldistance = None, pctdistance = 0.6, textprops={'fontsize': 14})
plt.legend( loc='best')
location = Counter(data['Location'])

most_common_location=location.most_common()
x,y= zip(*most_common_location)
x,y= list(x), list(y)

plt.figure(figsize=(15,10))
sns.barplot(x=x,y=y)
plt.xlabel('Most Common Location')
data['Sector'].unique()
plt.figure(figsize=(20,10))
sns.stripplot(x='Sector',y='Average Salary',data = data)
plt.title('Correlation between sectors and salaries')
plt.ylabel('Average Salary')
plt.xlabel('Sector')
plt.xticks(
    rotation=45,
    horizontalalignment='right')
min_sal = data.groupby(['Company Name', 'Job Title', 'Rating', 'Sector', 'Location'])['Min Salary'].mean().reset_index()
min_sal.sort_values(by=(['Rating', 'Min Salary']),ascending=False).head(10)
max_sal = data.groupby(['Company Name', 'Job Title', 'Rating', 'Sector', 'Location'])['Max Salary'].mean().reset_index()
max_sal.sort_values(by=(['Rating', 'Max Salary']),ascending=False).head(10)