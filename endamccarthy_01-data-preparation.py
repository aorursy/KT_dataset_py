import pandas as pd                # pandas is a python library used for data analysis

import numpy as np                 # numpy is a scientific computing library for python

import matplotlib.pyplot as plt    # matplotlib plots the charts

import seaborn as sns              # seaborn makes the charts look nice



sns.set(style="darkgrid")

%matplotlib inline



df = pd.read_csv('../input/bank1.csv') 
df.head()
df.tail()
df.describe()
# Find the number of nulls/NaNs in the dataset

df.apply(lambda x: sum(x.isnull()), axis=0)
df.subscribed[(df['poutcome'].isnull()) & (df['previous'] == 0)].count()
df['poutcome'].value_counts()
df['poutcome'].fillna('not applicable', inplace=True)

df['poutcome'].value_counts()
df['contact'].value_counts()
df['contact'].fillna('cellular', inplace=True)

df['contact'].value_counts()
df.job[df['education'].isnull()].count()
df.job[df['education'].isnull()].value_counts()
pivot_table_tertiary = df.pivot_table(values='education',

                       index=['job'],

                       aggfunc=lambda x: x.map({'tertiary':1, 'secondary':0, 'primary':0}).mean()) 

pivot_table_tertiary.plot(kind='bar')

plt.xlabel('Job')

plt.ylabel('Probability of Having Tertiary Education')

plt.title('Probability of Having Tertiary Education by Job')

plt.legend().set_visible(False) # we don't need the default legend
df.loc[((df['job'] == 'management') | 

        (df['job'] == 'self-employed')) & (df['education'].isnull()), 'education'] = 'tertiary'
pivot_table_secondary = df.pivot_table(values='education',

                       index=['job'],

                       aggfunc=lambda x: x.map({'tertiary':0, 'secondary':1, 'primary':0}).mean()) 

pivot_table_secondary.plot(kind='bar')

plt.xlabel('Job')

plt.ylabel('Probability of Having Secondary Education')

plt.title('Probability of Having Secondary Education by Job')

plt.legend().set_visible(False) # we don't need the default legend
df.loc[((df['job'] == 'admin.') | 

        (df['job'] == 'blue-collar') | 

        (df['job'] == 'services') | 

        (df['job'] == 'student') | 

        (df['job'] == 'technician') | 

        (df['job'] == 'unemployed')) & (df['education'].isnull()), 'education'] = 'secondary'
pivot_table_primary = df.pivot_table(values='education',

                       index=['job'],

                       aggfunc=lambda x: x.map({'tertiary':0, 'secondary':0, 'primary':1}).mean()) 

pivot_table_primary.plot(kind='bar')

plt.xlabel('Job')

plt.ylabel('Probability of Having Primary Education')

plt.title('Probability of Having Primary Education by Job')

plt.legend().set_visible(False) # we don't need the default legend
df.job[df['education'].isnull()].value_counts()
df.education[df['job'] == 'entrepreneur'].value_counts()
df.loc[((df['job'] == 'entrepreneur')) & (df['education'].isnull()), 'education'] = 'tertiary'
df.education[df['job'] == 'retired'].value_counts()
df.loc[((df['job'] == 'retired')) & (df['education'].isnull()), 'education'] = 'secondary'
df.loc[(df['education'].isnull()) & (df['job'].isnull())]
pivot_financial_tertiary = df.pivot_table(values='education', 

                                         index=((df['default'] == 'no') & 

                                         (df['housing'] == 'no') & 

                                         (df['loan'] == 'no') & 

                                         (df['balance'] >= 300) & 

                                         (df['balance'] <= 1900)),

                                         aggfunc=lambda x: x.map({'tertiary':1, 'secondary':0 , 'primary':0}).mean()) 

pivot_financial_tertiary.plot(kind='bar')

plt.xlabel('Not in default, no loans and balance between 300-1900')

plt.ylabel('Probability of Having Tertiary Education')

plt.title('Probability of Having Tertiary Education by Finacial Status')

plt.legend().set_visible(False) # we don't need the default legend
pivot_financial_secondary = df.pivot_table(values='education', 

                                         index=((df['default'] == 'no') & 

                                         (df['housing'] == 'no') & 

                                         (df['loan'] == 'no') & 

                                         (df['balance'] >= 300) & 

                                         (df['balance'] <= 1900)),

                                         aggfunc=lambda x: x.map({'tertiary':0, 'secondary':1, 'primary':0}).mean()) 

pivot_financial_secondary.plot(kind='bar')

plt.xlabel('Not in default, no loans and balance between 300-1900')

plt.ylabel('Probability of Having Secondary Education')

plt.title('Probability of Having Secondary Education by Finacial Status')

plt.legend().set_visible(False) # we don't need the default legend
df.loc[((df['job'].isnull())) & (df['education'].isnull()), 'education'] = 'secondary'
df.loc[df['job'].isnull()]
df['job'].fillna('not available', inplace=True)
table = np.round(df.pivot_table(values='age', 

                                index='job', 

                                columns='marital',

                                aggfunc=np.median), 0)

print(table)
# Define function to return an element of the pivot table

def get_element(x):

    return table.loc[x['job'], x['marital']]



# Replace missing values (only if there are any missing values)

if (df.subscribed[df['age'].isnull()].count() > 0):

    df['age'].fillna(df[df['age'].isnull()].apply(get_element, axis=1), inplace=True)
df.apply(lambda x: sum(x.isnull()), axis=0)
df['balance'].hist(bins=40)

plt.xlabel('Balance')

plt.ylabel('Number of People')
# Use a log transformation to decrease the impact of extreme values in column balance

df['balance_log'] = np.log(df['balance'] + abs(df['balance'].min()) + 1)
df['balance_log'].hist(bins=40)

plt.xlabel('log(Balance)')

plt.ylabel('Number of People')
df.subscribed[(df['duration'] <= 22) & (df['subscribed'] == 'yes')].count()
df['short_call'] = False 

df.loc[(df['duration'] <= 22), 'short_call'] = True