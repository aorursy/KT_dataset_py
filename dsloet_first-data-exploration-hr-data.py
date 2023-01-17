# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df =  pd.read_csv('../input/human-resources-data-set/core_dataset.csv')
df.head()
df.info()
df.columns
df.select_dtypes(include=['float64']).columns.values
df_isnull = (df.isnull().sum() / len(df))*100

df_isnull = df_isnull.drop(df_isnull[df_isnull ==0]).sort_values(ascending = False)[:30]

missing_data = pd.DataFrame({'Missing Ration' :df_isnull})

missing_data
df['Date of Termination'] = df['Date of Termination'].fillna("None")
df_isnull = (df.isnull().sum() / len(df))*100

df_isnull = df_isnull.drop(df_isnull[df_isnull ==0].index).sort_values(ascending = False)

missing_data = pd.DataFrame({'Missing Ration' :df_isnull})

missing_data



df.tail()
df = df[df.Position.notnull()]

df_isnull = (df.isnull().sum() / len(df))*100

df_isnull = df_isnull.drop(df_isnull[df_isnull ==0].index).sort_values(ascending = False)

missing_data = pd.DataFrame({'Missing Ration' :df_isnull})

missing_data

corrmat = df.corr()

plt.subplots(figsize=(4,4))

sns.heatmap(corrmat, vmax=0.9, square=True)
sns.distplot(df['Pay Rate'])
print(df['Pay Rate'].describe())

print("\nMedian of pay rate is: ", df['Pay Rate'].median(axis = 0))
sns.regplot( x = 'Zip', y = 'Pay Rate', data = df)
plt.figure(figsize = (16, 10))



sns.boxplot(x = 'Zip', y = 'Pay Rate', data = df)
sns.regplot( x = 'Age', y = 'Pay Rate', data = df)
df.rename(columns={

    'Pay Rate': 'PayRate',

    'Employee Name': 'EmployeeName',

    'Employee Number': 'EmployeeNumber',

    'Hispanic/Latino': 'HispLat',

    'Date of Hire': 'DateHire',

    'Days Employed': 'DaysEmployed',

    'Date of Termination': 'DateTerm',

    'Reason For Term': 'ReasonTerm',

    'Employment Status': 'EmployStatus',

    'Manager Name': 'ManagerName',

    'Employee Source': 'EmployeeSource',

    'Performance Score': 'PerformanceScore'



}, inplace=True)
df.head()
HispLat_map ={'No': 0, 'Yes': 1, 'no': 0, 'yes': 1}

df['HispLat'] = df['HispLat'].replace(HispLat_map)

df['HispLat']
pd.crosstab(df.CitizenDesc, df.Sex)
Sex_map ={'Female': 0, 'Male': 1, 'male': 0}

df['Sex'] = df['Sex'].replace(Sex_map)

pd.crosstab(df.CitizenDesc, df.Sex)
pd.crosstab(df.State, df.Sex)
sns.violinplot('HispLat', 'PayRate', data = df)
sns.violinplot('Sex', 'PayRate', data = df)
pd.crosstab(df.Sex.values, df.PerformanceScore.values)
g = sns.FacetGrid(df, col='Sex', row='MaritalDesc')

g.map(plt.hist, 'PayRate')
g = sns.FacetGrid(df, col='HispLat', row='MaritalDesc')

g.map(plt.hist, 'PayRate')
g = sns.FacetGrid(df, col='Sex', row='PerformanceScore')

g.map(plt.hist, 'PayRate')
df[['PerformanceScore', 'PayRate', 'Age']].groupby(['PerformanceScore'], 

as_index=False).mean()
PerfScore_map = {'90-day meets': 2, 'Exceeds': 3, 'Exceptional': 4, 'Fully Meets': 2, 'N/A- too early to review': 0,

                'Needs Improvement': 1, 'PIP': 1}



df['PerformanceScore'] = df['PerformanceScore'].replace(PerfScore_map)



df.head()
g = sns.FacetGrid(df, col='Sex', row='ManagerName')

g.map(plt.hist, 'PerformanceScore')
pd.crosstab(df.EmployeeSource.values, df.PerformanceScore.values) 
g = sns.FacetGrid(df, row='EmployeeSource')

g.map(plt.hist, 'PerformanceScore')
sns.boxplot(y = 'EmployeeSource', x = 'PerformanceScore', data = df)
sns.boxplot(y = 'ManagerName', x = 'PerformanceScore', data = df)
sns.boxplot(y = 'PayRate', x = 'PerformanceScore', data = df)
sns.boxplot(y = 'RaceDesc', x = 'PerformanceScore', data = df)
sns.boxplot(y = 'RaceDesc', x = 'PayRate', data = df)
sns.boxplot(y = 'MaritalDesc', x = 'PerformanceScore', data = df)
sns.boxplot(y = 'MaritalDesc', x = 'PayRate', data = df)
sns.boxplot(y = 'ReasonTerm', x = 'PerformanceScore', data = df)