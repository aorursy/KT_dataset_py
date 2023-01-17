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
import matplotlib as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv',index_col = 'Id',na_values=['Not Provided','Not provided','NAN'])
df['EmployeeName_Caps'] = df['EmployeeName'].apply(lambda x: str(x).upper())

df['JobTitle_Caps'] = df['JobTitle'].apply(lambda x: str(x).upper())

df.drop(['EmployeeName','JobTitle'], axis=1, inplace=True)

df.head()
df.columns
df.info()
df['JobTitle_Caps'].nunique()
df_titles = pd.read_excel('/kaggle/input/hourlyratesofpaybyclassificationandstepfy1920/Hourly-Rates-of-Pay-by-Classification-and-Step-FY19-20.xlsx', header=5)
df_titles.head()

police_titles = df_titles[df_titles['Union Code']==911]['Title']

police_titles
police_code_words = police_titles.apply(lambda x: x.upper().split()[0]).unique()

police_code_words = np.delete(police_code_words, 4) # Removing 'Assistant', since it's a generic word.

police_code_words
def is_police_job(job_title):

    for i in police_code_words:

        if i in job_title:

            return True

    return False



df_police = df[df['JobTitle_Caps'].apply(is_police_job)]

df_police

for i in df_police['JobTitle_Caps'].unique():

    print(i)
for i in df.columns:

    print('***************')

    print(df[i].value_counts())

    print('***************')

    print('\n'*2)
for i in df.columns:

    print('***************')

    print(df[i].describe())

    print('***************')

    print('\n'*2)
df[['BasePay','TotalPay','TotalPayBenefits']].plot.kde(xlim = (-10000, 600000))
df[df['JobTitle_Caps'].apply(lambda x : "POLICE" in x or 'SERGEANT' in x)]
job_year = df[df['JobTitle_Caps'].apply(lambda x : "POLICE" in x or 'SERGEANT' in x)][['JobTitle_Caps', 'Year']]



for i in df['Year'].unique():

    print('************** ' + str(i) + ' **************' )

    print(job_year[job_year['Year']==i]['JobTitle_Caps'].value_counts())

    print('***************' + '****' + '***************' )

    print('\n'*2)
df[df['JobTitle_Caps'] == 'CHIEF OF POLICE']
df[df['EmployeeName_Caps'] == 'Lorenzo Adamson'.upper()]
df[df['EmployeeName_Caps'].apply(lambda x: 'Richard Hastings'.upper() in x)]
df[df['EmployeeName_Caps'].apply(lambda x: 'Shaughn Ryan'.upper() in x)]

#sns.countplot(sorted([df['EmployeeName'][i][0] for i in range(len(df))]))
#sorted([_.split()[0] for _ in df['EmployeeName'] if _[0]=='J'], reverse = True)