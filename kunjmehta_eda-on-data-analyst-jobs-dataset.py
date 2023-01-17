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
dataset = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv', index_col = 0)

dataset.head(3)
dataset.shape
dataset.info()
dataset.isna().sum()
dataset.dropna(inplace = True)

dataset.isna().sum()
dataset.drop(['Job Title', 'Job Description', 'Founded', 'Headquarters', 'Industry', 'Competitors', 'Easy Apply'], inplace = True, axis = 1)

dataset.head(1)
dataset_q1 = dataset.copy()

dataset_q1 = dataset[['Sector','Location']]
dataset_q1['Location'].unique()
dataset_q1['Location'] = dataset_q1['Location'].str[-2:]

dataset_q1['Location'].unique()
dataset_q1['Sector'].unique()
dataset_q1 = dataset_q1.replace({'Sector': {'-1': 'Unknown'}})

dataset_q1['Sector'].unique()
dataset_q1 = dataset_q1.groupby(by = ['Location', 'Sector'], as_index = False).size()

dataset_q1
data_q1 = dataset_q1.set_index(['Location', 'Sector'])['size']

ax = data_q1.unstack().plot(kind='barh', legend = True, figsize = (10,15), stacked = True, cmap = 'tab20b')

ax.set_ylabel('States')

ax.set_xlabel('Number of companies')

ax.set_title('Graph showing presence of companies according to sector in each state')
dataset_q2 = dataset[['Revenue', 'Size', 'Salary Estimate', 'Sector']]
dataset_q2 = dataset_q2[dataset_q2['Sector'] != '-1']

len(dataset_q2)
dataset_q2['Revenue'].unique()

dataset_q2 = dataset_q2[dataset_q2['Revenue'] != 'Unknown / Non-Applicable']

len(dataset_q2)
dataset_q2['Size'].unique()

dataset_q2 = dataset_q2[dataset_q2['Size'] != 'Unknown']

len(dataset_q2)
dataset_q2['Sector'].unique()

dataset_q2 = dataset_q2[dataset_q2['Sector'] != 'Unknown']

len(dataset_q2)
# Getting median salary

salary_split = dataset['Salary Estimate'].str.split("-" , expand = True)

dataset_q2['Salary Estimate'] = (pd.to_numeric(salary_split[0].str.extract('(\d+)' , expand = False)) +  \

                                 pd.to_numeric(salary_split[1].str.extract('(\d+)' , expand = False)) ) / 2

dataset_q2['Salary Estimate']
dataset_q2['Size'].unique()

for i in range(len(dataset_q2)):

    if len(dataset_q2['Size'].iloc[i].split(' ')) > 2:

        dataset_q2['Size'].iloc[i] = dataset_q2['Size'].iloc[i].split(' ')[2]

    else:

        dataset_q2['Size'].iloc[i] = '10000+'

dataset_q2['Size'], dataset_q2['Size'].unique()
dataset_q2['Revenue'].unique()

dataset_q2['Revenue'] = dataset_q2['Revenue'].str[:-5]

dataset_q2['Revenue']
dataset_q2_1 = dataset_q2[['Revenue', 'Salary Estimate']]



for i in range(len(dataset_q2_1)):

    if dataset_q2_1['Salary Estimate'].iloc[i] <= 50:

        dataset_q2_1['Salary Estimate'].iloc[i] = 'Less than 50k'

    elif dataset_q2_1['Salary Estimate'].iloc[i] > 50 and dataset_q2_1['Salary Estimate'].iloc[i] <= 100:

        dataset_q2_1['Salary Estimate'].iloc[i] = '50k - 100k'

    else:

        dataset_q2_1['Salary Estimate'].iloc[i] = 'Greater than 100k'





dataset_q2_1.groupby(by = ['Revenue', 'Salary Estimate'], as_index = False).size().head(5)
data_q2_1 = dataset_q2_1.groupby(by = ['Revenue', 'Salary Estimate'], as_index = False).size()

data_q2_1 = data_q2_1.set_index(['Revenue', 'Salary Estimate'])['size']

ax = data_q2_1.unstack().plot(kind='bar', legend = True, figsize = (5,5), cmap = 'tab20b')

ax.set_ylabel('Number of companies')

ax.set_title('Salary estimates according to company revenue')
dataset_q2_2 = dataset_q2[['Size', 'Salary Estimate']]



for i in range(len(dataset_q2_2)):

    if dataset_q2_2['Salary Estimate'].iloc[i] <= 50:

        dataset_q2_2['Salary Estimate'].iloc[i] = 'Less than 50k'

    elif dataset_q2_2['Salary Estimate'].iloc[i] > 50 and dataset_q2_2['Salary Estimate'].iloc[i] <= 100:

        dataset_q2_2['Salary Estimate'].iloc[i] = '50k - 100k'

    else:

        dataset_q2_2['Salary Estimate'].iloc[i] = 'Greater than 100k'





dataset_q2_2.groupby(by = ['Size', 'Salary Estimate'], as_index = False).size().head(5)
data_q2_2 = dataset_q2_2.groupby(by = ['Size', 'Salary Estimate'], as_index = False).size()

data_q2_2 = data_q2_2.set_index(['Size', 'Salary Estimate'])['size']

ax = data_q2_2.unstack().plot(kind='bar', legend = True, figsize = (5,5), cmap = 'tab20b')

ax.set_xlabel('Size of company')

ax.set_ylabel('Number of companies')

ax.set_title('Salary estimates related to size of the company')
dataset_q3 = dataset[['Rating', 'Salary Estimate', 'Revenue']]

dataset_q3 = dataset_q3[dataset_q3['Rating'] != -1]

dataset_q3 = dataset_q3[dataset_q3['Revenue'] != 'Unknown / Non-Applicable']



salary_split = dataset_q3['Salary Estimate'].str.split("-" , expand = True)

dataset_q3['Salary Estimate'] = (pd.to_numeric(salary_split[0].str.extract('(\d+)' , expand = False)) +  \

                                 pd.to_numeric(salary_split[1].str.extract('(\d+)' , expand = False)) ) / 2

for i in range(len(dataset_q3)):

    if dataset_q3['Salary Estimate'].iloc[i] <= 50:

        dataset_q3['Salary Estimate'].iloc[i] = 'Less than 50k'

    elif dataset_q3['Salary Estimate'].iloc[i] > 50 and dataset_q3['Salary Estimate'].iloc[i] <= 100:

        dataset_q3['Salary Estimate'].iloc[i] = '50k - 100k'

    else:

        dataset_q3['Salary Estimate'].iloc[i] = 'Greater than 100k'

        

dataset_q3['Revenue'] = dataset_q3['Revenue'].str[:-5]

        

dataset_q3.head(3), len(dataset_q3)
dataset_q3['Rating'] = round(dataset_q3['Rating'])
data_q3 = dataset_q3.groupby(by = ['Rating', 'Salary Estimate'], as_index = False).size()

data_q3 = data_q3.set_index(['Rating', 'Salary Estimate'])['size']

ax = data_q3.unstack().plot(kind='barh', legend = True, figsize = (5,5), cmap = 'tab20b')

ax.set_xlim(0,400)

ax.set_ylabel('Ratings (rounded)')

ax.set_xlabel('Number of companies')

ax.set_title('Salary estimates for differently rated companies')
dataset_q4 = dataset[['Type of ownership', 'Salary Estimate']]

dataset_q4 = dataset_q4[(dataset_q4['Type of ownership'] != 'Unknown') & (dataset_q4['Type of ownership'] != '-1')]



salary_split = dataset_q4['Salary Estimate'].str.split("-" , expand = True)

dataset_q4['Salary Estimate'] = (pd.to_numeric(salary_split[0].str.extract('(\d+)' , expand = False)) +  \

                                 pd.to_numeric(salary_split[1].str.extract('(\d+)' , expand = False))) 

for i in range(len(dataset_q4)):

    if dataset_q4['Salary Estimate'].iloc[i] <= 50:

        dataset_q4['Salary Estimate'].iloc[i] = 'Less than 50k'

    elif dataset_q4['Salary Estimate'].iloc[i] > 50 and dataset_q4['Salary Estimate'].iloc[i] <= 100:

        dataset_q4['Salary Estimate'].iloc[i] = '50k - 100k'

    else:

        dataset_q4['Salary Estimate'].iloc[i] = 'Greater than 100k'

dataset_q4.head(3)
data_q4 = dataset_q4.groupby(by = ['Type of ownership', 'Salary Estimate'], as_index = False).size()

data_q4 = data_q4.set_index(['Type of ownership', 'Salary Estimate'])['size']

ax = data_q4.unstack().plot(kind='barh', legend = True, figsize = (5,5), cmap = 'tab20b')

ax.set_xlim(0,400)

ax.set_xlabel('Number of companies')

ax.set_title('Salary estimates for differently owned companies')