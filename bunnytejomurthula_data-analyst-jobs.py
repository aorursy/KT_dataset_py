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
df = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
df.head()
df.shape
df.info()
# remove the unnamed:0 column in the dataset



df.drop('Unnamed: 0', axis = 1, inplace = True)
# Missing data



df.isnull().sum()
df['Job Title'].value_counts()
df['Salary Estimate'].value_counts()
df['Rating'].value_counts()
df['Founded'].value_counts()
df['Location'].value_counts()
df['Size'].value_counts()
df['Headquarters'].value_counts()
df['Type of ownership'].value_counts()
df['Industry'].value_counts()
df['Sector'].value_counts()
df['Revenue'].value_counts()
df['Competitors'].value_counts()
df['Easy Apply'].value_counts()
df = df.replace(-1,np.nan)

df = df.replace(-1.0,np.nan)

df = df.replace('-1',np.nan)
df.isnull().sum()
# we can observe from the dataset that the company name is having \n rating at the end of the name annd it can be removed



df['Company Name'],_ = df['Company Name'].str.split('\n', 1). str
# we can observe that the column job title has both title and department in same column and they can be seperated



df['Job Title'], df['Department'] = df['Job Title'].str.split(',', 1).str
# We can observe from the salary estimate colmn that the salary is estimated by glassdoor at the end of estimated values and this can be removed.



df['Salary Estimate'],_ = df['Salary Estimate'].str.split('(', 1).str
df.head()
# In salary estimate we can observe the presence of $ and K at left and right of the value respectively and they can be removed



df['Minimum Salary Estimate'], df['Maximum Salary Estimate'] = df['Salary Estimate'].str.split('-').str



df['Minimum Salary Estimate'] = df['Minimum Salary Estimate'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')



df['Maximum Salary Estimate'] = df['Maximum Salary Estimate'].str.strip(' ').str.lstrip('$').str.rstrip('K').fillna(0).astype('int')
df.head()
# now we can drop the salary estimate column



df.drop(['Salary Estimate'], axis = 1, inplace = True)
# As we can see easy apply column has Nan values hey can be converted to bool False



df['Easy Apply'] = df['Easy Apply'].fillna(False).astype(bool)
df.head()
# True in easy apply represents that the company is hiring at present

# we can now plot the company wise hiring 



df_hiring = df[df['Easy Apply'] == True]

df_company = df_hiring.groupby('Company Name').count().reset_index()

df_company = df_company.sort_values('Easy Apply', ascending = True)

df_company
# We can now visualize the jobs company wise

import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize = (18, 7))

sns.barplot('Company Name', 'Easy Apply', data = df_company, )

plt.xticks(rotation = 90)

plt.show()
# We can visulaize location wise salary trends 

df_location = df.groupby('Location')[['Minimum Salary Estimate', 'Maximum Salary Estimate']].mean().sort_values(['Maximum Salary Estimate', 'Minimum Salary Estimate'],ascending = False).head(50)

df_location
df_location.plot.bar(stacked = True, figsize = (20,6))

plt.show()
df.isnull().sum()
# We can replace missing values in rating with mean value

df['Rating'].fillna(df['Rating'].mean(), inplace = True)
df.isnull().sum()
df.fillna('Unknown', inplace = True)
df_rating = df.groupby('Rating')['Company Name'].count()

df_rating = pd.DataFrame(df_rating).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_rating = df_rating.sort_values('Count of companies', ascending = False)

df_rating
plt.figure(figsize = (20,6))

sns.barplot(x = df_rating['Rating'],y = df_rating['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
# Top places where most companies headquarters are located



df_hq = df.groupby('Headquarters')['Company Name'].count()

df_hq = pd.DataFrame(df_hq).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_hq = df_hq.sort_values('Count of companies', ascending = False).head(30)

df_hq
plt.figure(figsize = (10,6))

sns.barplot(x = df_hq['Headquarters'],y = df_hq['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df['Size']
df_size = df.groupby('Size')['Company Name'].count()

df_size = pd.DataFrame(df_size).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_size = df_size.sort_values('Count of companies', ascending = False)

df_size
plt.figure(figsize = (10,6))

sns.barplot(x = df_size['Size'],y = df_size['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_found = df.groupby('Founded')['Company Name'].count()

df_found = pd.DataFrame(df_found).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_found = df_found.sort_values('Count of companies', ascending = False).head(50)

df_found
plt.figure(figsize = (15,5))

sns.barplot(x = df_found['Founded'],y = df_found['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_owner = df.groupby('Type of ownership')['Company Name'].count()

df_owner = pd.DataFrame(df_owner).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_owner = df_owner.sort_values('Count of companies', ascending = False)





plt.figure(figsize = (10,5))

sns.barplot(x = df_owner['Type of ownership'],y = df_owner['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_industry = df.groupby('Industry')['Company Name'].count()

df_industry = pd.DataFrame(df_industry).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_industry = df_industry.sort_values('Count of companies', ascending = False)





plt.figure(figsize = (20,5))

sns.barplot(x = df_industry['Industry'],y = df_industry['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_sector = df.groupby('Sector')['Company Name'].count()

df_sector = pd.DataFrame(df_sector).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_sector = df_sector.sort_values('Count of companies', ascending = False)





plt.figure(figsize = (20,5))

sns.barplot(x = df_sector['Sector'],y = df_sector['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_revenue = df.groupby('Revenue')['Company Name'].count()

df_revenue = pd.DataFrame(df_revenue).rename(columns = {'Company Name': 'Count of companies'}).reset_index()

df_revenue = df_revenue.sort_values('Count of companies', ascending = False)





plt.figure(figsize = (20,5))

sns.barplot(x = df_revenue['Revenue'],y = df_revenue['Count of companies'])

plt.xticks(rotation = 90)

plt.show()
df_ind = df.groupby('Industry')[['Minimum Salary Estimate', 'Maximum Salary Estimate']].mean().sort_values(['Maximum Salary Estimate', 'Minimum Salary Estimate'],ascending = False)

df_ind = df_ind.rename(columns = {'Minimum Salary Estimate' : 'Average Minimum Salary', 'Maximum Salary Estimate' : 'Average Maximum Salary'})

df_ind
df_ind.plot.bar(stacked = True, figsize = (25,5))

plt.show()
df_open = pd.DataFrame(df[df['Easy Apply'] == True]['Job Title'].value_counts()).rename(columns={'Job Title':'Number of openings'}).reset_index().rename(columns={'index':'Job Title'})

df_open
plt.figure(figsize = (25, 5))

sns.barplot(df_open['Job Title'], df_open['Number of openings'])

plt.xticks(rotation = 90)

plt.show()