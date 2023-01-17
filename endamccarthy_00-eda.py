import pandas as pd                # pandas is a python library used for data analysis

import numpy as np                 # numpy is a scientific computing library for python

import matplotlib.pyplot as plt    # matplotlib plots the charts

import seaborn as sns              # seaborn makes the charts look nice

sns.set(style="darkgrid")
df = pd.read_csv('../input/bank.csv')
df.head(5)  
# defaults to 5 if no value is given

df.tail()
# prints a statistical summary for all numerical attributes

df.describe()
df['education'].count()
df['education'].value_counts()
df['poutcome'].count()
df.subscribed[df['poutcome'].isnull()].count()
df['poutcome'].value_counts()
df.subscribed[(df['poutcome'] == 'other') & (df['subscribed'] == 'yes')].count()
df.subscribed[(df['poutcome'] == 'other') & (df['subscribed'] == 'no')].count()
df.subscribed[(df['poutcome'].isnull()) & (df['previous'] == 0)].count()
df['age'].hist(bins=20)

plt.xlabel('Age')

plt.ylabel('Number of People')
df['balance'].hist(bins=40)

plt.xlabel('Balance')

plt.ylabel('Number of People')
df.subscribed[df['balance'] > 10000].count()
df.balance[df['balance'] < 10000].hist(bins=40)

plt.xlabel('Balance')

plt.ylabel('Number of People')
df[df['balance'] < 10000].boxplot(column='balance')
df[df['balance'] < 10000].boxplot(column='balance', by='loan')

plt.title('Boxplot of Balance grouped by Personal Loan Status')

plt.suptitle("") # get rid of the automatic 'Boxplot grouped by group_by_column_name' title
sns.regplot(x='balance', y='age', data=df[df['balance'] < 10000])
# boxplot

sns.boxplot(x=df.balance[df['balance'] < 10000])
# violin plot

sns.violinplot(x=df.balance[df['balance'] < 10000])
# strip plot

sns.stripplot(x=df.balance[df['balance'] < 10000], jitter=True)
# swarm plot

sns.swarmplot(x=df.balance[df['balance'] < 10000])
frequency_table = df['marital'].value_counts(ascending=True)

print('Frequency Table for Marital Status:') 

print(frequency_table)
frequency_table.plot(kind='bar')

plt.xlabel('Marital Status')

plt.ylabel('Number of People')

plt.title('People by Marital Status')
pivot_table = df.pivot_table(values='subscribed',

                       index=['marital'],

                       aggfunc=lambda x: x.map({'yes':1, 'no':0}).mean()) 

print(pivot_table)
pivot_table.plot(kind='bar')

plt.xlabel('Marital Status')

plt.ylabel('Probability of Subscribing')

plt.title('Probability of Subscribing by Marital Status')

plt.legend().set_visible(False) # we don't need the default legend
frequency_table_edu = df['education'].value_counts(ascending=True)

frequency_table_edu.plot(kind='bar')

plt.xlabel('Education')

plt.ylabel('Number of People')

plt.title('People by Education')
pivot_table_edu = df.pivot_table(values='subscribed',

                       index=['education'],

                       aggfunc=lambda x: x.map({'yes':1, 'no':0}).mean()) 

pivot_table_edu.plot(kind='bar')

plt.xlabel('Education')

plt.ylabel('Probability of Subscribing')

plt.title('Probability of Subscribing by Education')

plt.legend().set_visible(False) # we don't need the default legend
stacked_chart = pd.crosstab(df['marital'], df['subscribed'])

stacked_chart.plot(kind='bar', stacked=True, color=['indianred', 'palegreen'])

plt.xlabel('Marital Status')

plt.ylabel('Number of People')
stacked_chart_gender = pd.crosstab([df['marital'], df['education']], df['subscribed'])

stacked_chart_gender.plot(kind='bar', stacked=True, color=['indianred', 'palegreen'])

plt.ylabel('Number of Applicants')