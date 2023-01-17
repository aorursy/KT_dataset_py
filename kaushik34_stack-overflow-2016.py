import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('white')

stack = pd.read_csv('../input/2016 Stack Overflow Survey Responses.csv')
df = stack['occupation_group'].value_counts()

addition = sum(df)

# print(addition)

developer_occupation_perc = round((df * 100) / addition)

# print(developer_occupation_perc)

df.head(10).plot.barh(figsize=(40,30),width=0.4)

plt.xticks(size = 40)

plt.yticks(size = 40)

plt.ylabel('Occupation',size=40)

plt.xlabel('People Fall in the Range',size=40)
stack['employment_status'].value_counts()
stack.info()
stack['tech_do'].value_counts().head(20)
df_t = stack.dropna()
stack.info()
stack['rep_range'].value_counts()
stack['rep_range'].value_counts().plot.bar()
stack['visit_frequency'].value_counts().plot.barh()
dfs = pd.crosstab(stack['occupation_group'],stack['agree_diversity'])

# addition = sum(df)

# # print(addition)

# developer_occupation_perc = round((df * 100) / addition)

# # print(developer_occupation_perc)

dfs.head().plot.barh(figsize=(30,30),width=0.4)

plt.xticks(size = 70)

plt.yticks(size = 70)

plt.ylabel('Occupation',size=70)

plt.xlabel('People Fall in the Range',size=70)
nothing = stack['why_stack_overflow'].value_counts().head(10)

add = sum(nothing)

nothing_duff = round((nothing * 100) / add)

nothing.plot.bar(figsize=(30,30),width=0.5)

plt.xticks(size = 70)

plt.yticks(size = 70)

plt.ylabel('Occupation',size=60)

plt.xlabel('People Fall in the Range',size=60)
stack.head()
def child(passenger):

    age_midpoint,gender = passenger

    if age_midpoint<20:

        return 'youngstar'

    else:

        return gender

stack['person'] = stack[['age_midpoint', 'gender']].apply(child,axis=1)
stack.person.head()
bar = stack.commit_frequency.value_counts()

bar.plot.bar()
stack.job_satisfaction.value_counts()
notf=pd.crosstab(stack.job_satisfaction,stack.commit_frequency)

notf.plot.bar(figsize=(30,30),width=0.9)

plt.xticks(size = 70)

plt.yticks(size = 70)

plt.ylabel('Occupation',size=60)

plt.xlabel('People Fall in the Range',size=60)

plt.figtext(0.9,0.9,0.9)
full = pd.crosstab(stack.remote,stack.job_satisfaction)

full['I love my job'].plot.bar()
remo = pd.crosstab(stack.country,stack.remote)

remo.head().plot.barh(figsize=(20,30),width=1.5)

plt.xticks(size = 40)

plt.yticks(size = 30)

plt.ylabel('age',size=50)