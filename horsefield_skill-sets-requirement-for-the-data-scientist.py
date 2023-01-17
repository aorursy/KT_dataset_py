# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df_multiple = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)
# Q12

df_multiple['Q12_Part_1'].replace('Twitter (data science influencers)', 'Twitter', inplace=True)

df_multiple['Q12_Part_2'].replace('Hacker News (https://news.ycombinator.com/)', 'Hacker', inplace=True)

df_multiple['Q12_Part_3'].replace('Reddit (r/machinelearning, r/datascience, etc)', 'Reddit', inplace=True)

df_multiple['Q12_Part_4'].replace('Kaggle (forums, blog, social media, etc)', 'Kaggle', inplace=True)

df_multiple['Q12_Part_5'].replace('Course Forums (forums.fast.ai, etc)', 'Course', inplace=True)

df_multiple['Q12_Part_6'].replace('YouTube (Cloud AI Adventures, Siraj Raval, etc)', 'YouTube', inplace=True)

df_multiple['Q12_Part_7'].replace('Podcasts (Chai Time Data Science, Linear Digressions, etc)', 'Podcasts', inplace=True)

df_multiple['Q12_Part_8'].replace('Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)', 'Blogs', inplace=True)

df_multiple['Q12_Part_9'].replace('Journal Publications (traditional publications, preprint journals, etc)', 'Journal', inplace=True)

df_multiple['Q12_Part_10'].replace('Slack Communities (ods.ai, kagglenoobs, etc)', 'Slack', inplace=True)

df_multiple['Q12_Part_11'].replace('None', 'None', inplace=True)

df_multiple['Q12_Part_12'].replace('Other', 'Other', inplace=True)



# Q24

df_multiple['Q24_Part_1'].replace('Linear or Logistic Regression', 'Line or Logi', inplace=True)

df_multiple['Q24_Part_2'].replace('Decision Trees or Random Forests', 'DTorRF', inplace=True)

df_multiple['Q24_Part_3'].replace('Gradient Boosting Machines (xgboost, lightgbm, etc)', 'Gbm', inplace=True)

df_multiple['Q24_Part_4'].replace('Bayesian Approaches', 'Bayesian', inplace=True)

df_multiple['Q24_Part_5'].replace('Evolutionary Approaches', 'Evolutionary', inplace=True)

df_multiple['Q24_Part_6'].replace('Dense Neural Networks (MLPs, etc)', 'Mlp', inplace=True)

df_multiple['Q24_Part_7'].replace('Convolutional Neural Networks', 'Cnn', inplace=True)

df_multiple['Q24_Part_8'].replace('Generative Adversarial Networks', 'Generative', inplace=True)

df_multiple['Q24_Part_9'].replace('Recurrent Neural Networks', 'RNN', inplace=True)

df_multiple['Q24_Part_10'].replace('Transformer Networks (BERT, gpt-2, etc)', 'Transformer', inplace=True)

df_multiple['Q24_Part_11'].replace('None', 'None', inplace=True)

df_multiple['Q24_Part_12'].replace('Other', 'Other', inplace=True)
# About Q12 Who/what are your favorite media sources that report on data science topics?

list_q12 = ['Q12_Part_1', 'Q12_Part_2', 'Q12_Part_3', 'Q12_Part_4', 'Q12_Part_5', 'Q12_Part_6', 'Q12_Part_7', 'Q12_Part_8']

df_mult_12 = df_multiple.loc[1:, list_q12]



df_mult_12_ = df_mult_12.loc[:, list_q12].fillna('')

df_mult_12_['Q12_matome'] = ''

for cols in list_q12:

    df_mult_12_['Q12_matome'] = df_mult_12_['Q12_matome'].str.cat(df_mult_12_[cols])

df_mult_12['Q12_matome'] = df_mult_12_['Q12_matome']

df_mult_12.replace('', 'NaN', inplace=True)





# About Q18 What programming languages do you use on a regular basis?

list_q18 = ['Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12']

df_mult_18 = df_multiple.loc[1:, list_q18]



df_mult_18_ = df_mult_18.loc[:, list_q18].fillna('')

df_mult_18_['Q18_matome'] = ''

for cols in list_q18:

    df_mult_18_['Q18_matome'] = df_mult_18_['Q18_matome'].str.cat(df_mult_18_[cols])

df_mult_18['Q18_matome'] = df_mult_18_['Q18_matome']

df_mult_18.replace('', 'NaN', inplace=True)





# About Q24 Which of the following ML algorithms do you use on a regular basis?

list_q24 = ['Q24_Part_1', 'Q24_Part_2', 'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5', 'Q24_Part_6', 'Q24_Part_7', 'Q24_Part_8', 'Q24_Part_9', 'Q24_Part_10', 'Q24_Part_11', 'Q24_Part_12']

df_mult_24 = df_multiple.loc[1:, list_q24]



df_mult_24_ = df_mult_24.loc[:, list_q24].fillna('')

df_mult_24_['Q24_matome'] = ''

for cols in list_q24:

    df_mult_24_['Q24_matome'] = df_mult_24_['Q24_matome'].str.cat(df_mult_24_[cols])

df_mult_24['Q24_matome'] = df_mult_24_['Q24_matome']

df_mult_24.replace('', 'NaN', inplace=True)
df_mult_q = df_multiple.loc[1:, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]

df_mult_q['Q12_matome'] = df_mult_12_['Q12_matome']

df_mult_q['Q18_matome'] = df_mult_18_['Q18_matome']

df_mult_q['Q24_matome'] = df_mult_24_['Q24_matome']

df_mult_q = df_mult_q.replace('', 'Unknown')



list_top20_q12 = df_mult_q['Q12_matome'].value_counts().head(20).index.to_list()

list_top20_q18 = df_mult_q['Q18_matome'].value_counts().head(20).index.to_list()

list_top20_q24 = df_mult_q['Q24_matome'].value_counts().head(20).index.to_list()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

ax1.bar(df_mult_q.loc[:, 'Q2'].unique(), df_mult_q.loc[:, 'Q2'].value_counts())

ax2.pie(df_mult_q.loc[:, 'Q2'].value_counts(), labels=df_mult_q.loc[:, 'Q2'].unique(),

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 3, 'edgecolor':"white"})

ax3.bar(df_multiple.loc[1:, 'Q5'].dropna().unique(), df_multiple.loc[1:, 'Q5'].dropna().value_counts())

ax4.pie(df_multiple.loc[1:, 'Q5'].dropna().value_counts(), labels=df_multiple.loc[1:, 'Q5'].dropna().unique(),

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 3, 'edgecolor':"white"})

ax1.tick_params(axis='x', labelrotation=0)

ax3.tick_params(axis='x', labelrotation=90)

ax2.set_title('Q2   Ratio')

ax4.set_title('Q5   Ratio')

ax1.set_title('Q2   What is your gender?')

ax3.set_title('Q5   Select the title most similar to your current role (or most recent title if retired)')
multiple_q12_1 = df_multiple.loc[1:, 'Q12_Part_1'].value_counts(dropna=False)

multiple_q12_2 = df_multiple.loc[1:, 'Q12_Part_2'].value_counts(dropna=False)

multiple_q12_3 = df_multiple.loc[1:, 'Q12_Part_3'].value_counts(dropna=False)

multiple_q12_4 = df_multiple.loc[1:, 'Q12_Part_4'].value_counts(dropna=False)

multiple_q12_5 = df_multiple.loc[1:, 'Q12_Part_5'].value_counts(dropna=False)

multiple_q12_6 = df_multiple.loc[1:, 'Q12_Part_6'].value_counts(dropna=False)

multiple_q12_7 = df_multiple.loc[1:, 'Q12_Part_7'].value_counts(dropna=False)

multiple_q12_8 = df_multiple.loc[1:, 'Q12_Part_8'].value_counts(dropna=False)



list_mult = [multiple_q12_1, multiple_q12_2, multiple_q12_3, multiple_q12_4, multiple_q12_5, multiple_q12_6, multiple_q12_7, multiple_q12_8]

df_mult = pd.DataFrame(list_mult).T

df_mult['result'] = df_mult.sum(axis=1)

df_mult.rename(index={'Twitter (data science influencers)': 'Twitter',

                     'Hacker News (https://news.ycombinator.com/)': 'Hacker',

                     'Reddit (r/machinelearning, r/datascience, etc)': 'Reddit',

                     'Kaggle (forums, blog, social media, etc)': 'Kaggle',

                     'Course Forums (forums.fast.ai, etc)': 'Course',

                     'YouTube (Cloud AI Adventures, Siraj Raval, etc)': 'YouTube',

                     'Podcasts (Chai Time Data Science, Linear Digressions, etc)': 'podcasts',

                     'Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)': 'Blogs'}, inplace=True)

df_mult_sort = df_mult.loc[:, 'result'].sort_values()

df_mult_sort = df_mult_sort.drop(np.nan, axis=0)





multiple_q18_1 = df_multiple.loc[1:, 'Q18_Part_1'].value_counts(dropna=False)

multiple_q18_2 = df_multiple.loc[1:, 'Q18_Part_2'].value_counts(dropna=False)

multiple_q18_3 = df_multiple.loc[1:, 'Q18_Part_3'].value_counts(dropna=False)

multiple_q18_4 = df_multiple.loc[1:, 'Q18_Part_4'].value_counts(dropna=False)

multiple_q18_5 = df_multiple.loc[1:, 'Q18_Part_5'].value_counts(dropna=False)

multiple_q18_6 = df_multiple.loc[1:, 'Q18_Part_6'].value_counts(dropna=False)

multiple_q18_7 = df_multiple.loc[1:, 'Q18_Part_7'].value_counts(dropna=False)

multiple_q18_8 = df_multiple.loc[1:, 'Q18_Part_8'].value_counts(dropna=False)

multiple_q18_9 = df_multiple.loc[1:, 'Q18_Part_9'].value_counts(dropna=False)

multiple_q18_10 = df_multiple.loc[1:, 'Q18_Part_10'].value_counts(dropna=False)

multiple_q18_11 = df_multiple.loc[1:, 'Q18_Part_11'].value_counts(dropna=False)

multiple_q18_12 = df_multiple.loc[1:, 'Q18_Part_12'].value_counts(dropna=False)



list_q18 = [multiple_q18_1, multiple_q18_2, multiple_q18_3, multiple_q18_4, multiple_q18_5, multiple_q18_6,multiple_q18_7, multiple_q18_8, multiple_q18_9, multiple_q18_10, multiple_q18_11, multiple_q18_12]

df_q18 = pd.DataFrame(list_q18).T

df_q18['result'] = df_q18.sum(axis=1)

df_q18_sort = df_q18.loc[:, 'result'].sort_values()

df_q18_sort = df_q18_sort.drop(np.nan, axis=0)





multiple_q24_1 = df_multiple.loc[1:, 'Q24_Part_1'].value_counts(dropna=False)

multiple_q24_2 = df_multiple.loc[1:, 'Q24_Part_2'].value_counts(dropna=False)

multiple_q24_3 = df_multiple.loc[1:, 'Q24_Part_3'].value_counts(dropna=False)

multiple_q24_4 = df_multiple.loc[1:, 'Q24_Part_4'].value_counts(dropna=False)

multiple_q24_5 = df_multiple.loc[1:, 'Q24_Part_5'].value_counts(dropna=False)

multiple_q24_6 = df_multiple.loc[1:, 'Q24_Part_6'].value_counts(dropna=False)

multiple_q24_7 = df_multiple.loc[1:, 'Q24_Part_7'].value_counts(dropna=False)

multiple_q24_8 = df_multiple.loc[1:, 'Q24_Part_8'].value_counts(dropna=False)

multiple_q24_9 = df_multiple.loc[1:, 'Q24_Part_9'].value_counts(dropna=False)

multiple_q24_10 = df_multiple.loc[1:, 'Q24_Part_10'].value_counts(dropna=False)

multiple_q24_11 = df_multiple.loc[1:, 'Q24_Part_11'].value_counts(dropna=False)

multiple_q24_12 = df_multiple.loc[1:, 'Q24_Part_12'].value_counts(dropna=False)



list_q24 = [multiple_q24_1, multiple_q24_2, multiple_q24_3, multiple_q24_4, multiple_q24_5, multiple_q24_6,

            multiple_q24_7, multiple_q24_8, multiple_q24_9, multiple_q24_10, multiple_q24_11, multiple_q24_12]

df_q24 = pd.DataFrame(list_q24).T

df_q24['result'] = df_q24.sum(axis=1)

df_q24_sort = df_q24.loc[:, 'result'].sort_values()

df_q24_sort = df_q24_sort.drop(np.nan, axis=0)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

ax1.barh(df_mult_sort.index, df_mult_sort)

ax2.barh(df_q18_sort.index, df_q18_sort)

ax3.barh(df_q24_sort.index, df_q24_sort)

ax4.pie(df_q18_sort, labels=df_q18_sort.index,

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 3, 'edgecolor':"white"})

ax1.set_title('Q12  What is favorite media source to get data science info.')

ax2.set_title('Q18  What programming languages do you use on a regular basis?')

ax3.set_title('Q24  Which of the following ML algorithms do you use on a regular basis?')
df_crosstab_q5_q12 = pd.crosstab(df_mult_q.loc[1:,'Q5'], df_mult_q.loc[1:, 'Q12_matome'])

a512 = df_crosstab_q5_q12.loc[:,list_top20_q12]



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))

left = 0

for col in list_top20_q12:

    ax1.barh(a512.index, a512.loc[:, col], left=left)

    left += a512.loc[:, col]

ax1.legend(list_top20_q12, bbox_to_anchor=(3, 1), loc='upper right')

ax2.pie(a512.T.loc[:, 'Data Scientist'], labels=a512.T.loc[:, 'Data Scientist'].index,

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 3, 'edgecolor':"white"})

ax1.set_title('Q5 x Q12 Favorite media sources that report on data science topics each roll')

ax2.set_title('Q12 Favorite media sources ratio of Data Scientist')
df_crosstab_q18_q5 = pd.crosstab(df_mult_q.loc[1:, 'Q18_matome'], df_mult_q.loc[1:, 'Q5'])

a185 = df_crosstab_q18_q5.loc[list_top20_q18, :].T



col_name_q185 = df_crosstab_q18_q5.loc[list_top20_q18, :].T.columns

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

left = 0

for cols in col_name_q185:

    ax1.barh(a185.index, a185.loc[:, cols].T, left=left)

    left += a185.loc[:, cols].T

ax1.legend(col_name_q185, bbox_to_anchor=(3, 1), loc='upper right')

ax2.pie(a185.T.loc[:, 'Data Scientist'], labels=a185.T.loc[:, 'Data Scientist'].T.index,

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 2.8, 'edgecolor':"white"})

ax1.set_title('Q5 x Q18 Using programming languages each roll')

ax2.set_title('Q18 Using programming languages ratio of Data Scientist')
df_crosstab_q5_q24 = pd.crosstab(df_mult_q.loc[1:,'Q5'], df_mult_q.loc[1:, 'Q24_matome'])

a524 = df_crosstab_q5_q24.loc[:,list_top20_q24]



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))

left = 0

for col in list_top20_q24:

    ax1.barh(a524.index, a524.loc[:, col], left=left)

    left += a524.loc[:, col]

ax1.legend(list_top20_q24, bbox_to_anchor=(3.3, 1), loc='upper right')

ax2.pie(a524.T.loc[:, 'Data Scientist'], labels=a524.T.loc[:, 'Data Scientist'].index,

        startangle=90, autopct='%1.1f%%',

        wedgeprops={'linewidth': 3, 'edgecolor':"white"})

ax1.set_title('Q5 x Q24 The following ML algorithms each roll')

ax2.set_title('Q24 ML algorithms ratio of Data Scientist')
df_crosstab_q18_q24 = pd.crosstab(df_mult_q.loc[(df_mult_q['Q5']=='Data Scientist'),'Q18_matome'], df_mult_q.loc[(df_mult_q['Q5']=='Data Scientist'), 'Q24_matome'])

a1824 = df_crosstab_q18_q24.loc[:,list_top20_q24]

a1824['sum'] = a1824.sum(axis=1)

a_1824_datascientist = a1824.sort_values('sum', ascending=False).head(10)



a_1824_datascientist_t = a_1824_datascientist .T

a_1824_datascientist_t['sum'] = a_1824_datascientist_t .sum(axis=1)

df = a_1824_datascientist_t .sort_values('sum', ascending=False).iloc[1:11, 0:10]

df.style.background_gradient(cmap='Blues',

                             subset=pd.IndexSlice[:'Line or Logi', ['PythonRSQL', 'Python', 'PythonSQL', 'PythonR']])
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

plt.subplots_adjust(wspace = 0.4, hspace=3)

ax1.barh(a_1824_datascientist.loc[:, 'sum'].index, a_1824_datascientist.loc[:, 'sum'])

ax2.barh(a_1824_datascientist.T.loc[:, 'PythonRSQL'].drop('sum').head(10).index, a_1824_datascientist.T.loc[:, 'PythonRSQL'].drop('sum').head(10))

ax1.set_title('Ranking of Programming languages that Data Scientist use')

ax2.set_title('Ranking of ML algorithms using Python, R and SQL(Data Scientist)')