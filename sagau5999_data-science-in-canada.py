# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



COUNTRY = 'Canada'



df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv',header=1)

other_text_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

questions_only = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

survey_schema = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

####################################################### Country

Q_country = 'In which country do you currently reside?'

Q = Q_country

df[Q] = df[Q].replace(

        'United Kingdom of Great Britain and Northern Ireland'

        ,'United Kingdom')

df[Q] = df[Q].replace(

        'United States of America'

        ,'USA')



p = sns.countplot(data=df

                  ,x = Q

                  ,order = df[Q].value_counts().iloc[:10].index

                  ,palette = ['red' if (x == COUNTRY) else 'grey' for x in df[Q].value_counts().iloc[:10].index]

                  )

p.set_xticklabels(p.get_xticklabels(), rotation=40, ha="right")

p = p.set(ylabel = 'Count')
####################################################### Gender

Q_gender = 'What is your gender? - Selected Choice'

Q = Q_gender

df['Female'] = [1 if i == 'Female' else 0 for i in df[Q]]



fig, (p1, p2) = plt.subplots(1, 2)

fig.set_figwidth(15)

############################# Most popular countries

p1 = sns.barplot(x = Q_country, y = 'Female', data = df

            ,palette = ['red' if (x == COUNTRY) else 'grey' for x in df[Q_country].value_counts().iloc[:10].index]

            ,order = df[Q_country].value_counts().iloc[:10].index

            ,ci = None

            ,ax=p1

            )

p1.set_xticklabels(p1.get_xticklabels(), rotation=40, ha="right")

p1.set(title='Percent of females amongts the most popular countries on Kaggle')

p1.set(ylabel = 'Percentage of females')

############################# All countries

data = df.groupby(Q_country)['Female'].agg(['sum','count'])

data.columns = ['s','nb']

data = data[data.nb >= 100]

data = data['s'] / data['nb']

data.name = 'Percentage of females'

data = data.sort_values(ascending=False).iloc[0:10]



p2 = sns.barplot(x = data.index

                ,y = data

                ,palette = ['red' if (x == COUNTRY) else 'grey' for x in data.index]

                ,ci = None)

p2.set(title='Countries with the highest percentage of females on Kaggle')

p2 = p2.set_xticklabels(p2.get_xticklabels(), rotation=40, ha="right")
############################# Salary

Q_salary = 'What is your current yearly compensation (approximate $USD)?'

df['Country'] = [(COUNTRY if i == COUNTRY else ('Top 10 countries' if i in df[Q_country].value_counts().iloc[:10].index else 'Other Countries')) for i in df[Q_country]]

data = df.pivot_table(index='Country', columns=Q_salary

            ,aggfunc='size').fillna(0)

data.columns = [1000, 2000, 15000, 125000,

       150000, 20000, 200000, 3000,

       25000, 250000, 30000,300000,

       4000, 40000,500000, 5000,

       50000, 7500,60000,70000,

       10000, 80000, 90000, 100000,

       1000000]

data = data.reindex(sorted(data.columns), axis=1)

df_sum = data.sum(axis=1)

for col in data.columns:

    data[col] = data[col] / df_sum

data.plot(kind='bar', stacked=True,figsize=(8,15)

      ,title='Salaries'

      ,legend=True)

plt.subplots_adjust(right=2)
Q_educ = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'

df[Q_educ] = df[Q_educ].replace(

        'Some college/university study without earning a bachelor’s degree'

        ,'University without degree')

data = df.pivot_table(index='Country', columns=Q_educ

            ,aggfunc='size').fillna(0)

df_sum = data.sum(axis=1)

for col in data.columns:

    data[col] = data[col] / df_sum



plt.figure(figsize=(8, 6))

data.plot(kind='bar', stacked=True#,figsize=(8,15)

      ,title='Education'

      ,legend=False)

ax = plt.subplot(111)

box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width*0.65, box.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
from matplotlib.gridspec import GridSpec



Q_size = 'What is the size of the company where you are employed?'

Q_nb_ds = 'Approximately how many individuals are responsible for data science workloads at your place of business?'

Q_incorporate = 'Does your current employer incorporate machine learning methods into their business?'



fig = plt.figure(constrained_layout=True)

fig.set_figwidth(20)

fig.set_figheight(10)

gs = fig.add_gridspec(2, 2)

p1 = fig.add_subplot(gs[0, 0])

p2 = fig.add_subplot(gs[0, 1])

p3 = fig.add_subplot(gs[1, :])



############################# size

data = df.pivot_table(index='Country', columns=Q_size

            ,aggfunc='size').fillna(0)

df_sum = data.sum(axis=1)

for col in data.columns:

    data[col] = data[col] / df_sum



data.plot(kind='bar', stacked=True#,figsize=(8,15)

      ,title='Size of the company'

      ,legend=False

      ,ax = p1)

box = p1.get_position()

p1.set_position([box.x0, box.y0, box.width*0.65, box.height])

p1.legend(loc='center left', bbox_to_anchor=(1, 0.5))



############################# nb_ds

data = df.pivot_table(index='Country', columns=Q_nb_ds

            ,aggfunc='size').fillna(0)

df_sum = data.sum(axis=1)

for col in data.columns:

    data[col] = data[col] / df_sum



data.plot(kind='bar', stacked=True#,figsize=(8,15)

      ,title='Number of individuals responsible for data science'

      ,legend=False

      ,ax = p2)

box = p2.get_position()

p2.set_position([box.x0, box.y0, box.width*0.65, box.height])

p2.legend(loc='center left', bbox_to_anchor=(1, 0.5))



############################# incorporate

data = df.pivot_table(index='Country', columns=Q_incorporate

            ,aggfunc='size').fillna(0)

df_sum = data.sum(axis=1)

for col in data.columns:

    data[col] = data[col] / df_sum



data.plot(kind='bar', stacked=True#,figsize=(8,15)

      ,title='Incorporation of machine learning'

      ,legend=False

      ,ax = p3)

box = p3.get_position()

p3.set_position([box.x0, box.y0, box.width*0.65, box.height])

p3.legend(loc='center left', bbox_to_anchor=(1, 0.5))