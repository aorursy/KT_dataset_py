# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

df.columns=df.iloc[0]

df=df.drop([0])
col_to_analyze=[

 'What is your age (# years)?',

 'What is your gender? - Selected Choice',

 'In which country do you currently reside?',

 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',

 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice',

 'What is the size of the company where you are employed?']

df_to_analyze=df[col_to_analyze]

df_to_analyze.rename(columns={'What is your age (# years)?':'age',

                              'What is your gender? - Selected Choice':'gender',

                             'In which country do you currently reside?':'current_country',

                             'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'education_level',

                             'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'job_title',

                             'What is the size of the company where you are employed?':'company_size'},inplace=True)

def explore_columns(df):

    print('number of applicants in survey:\n',df.shape[0])

    print('number of selected questions:\n',df.shape[1])

    print('-'*50,'\n')

    for i in df.columns:

        print('column name: ',i)

        print('number of categories in this column:')

        print(df[i].value_counts())

        print('-'*50,'\n')

    print('number of missing values in each column: ')

    print(df.isnull().sum())

explore_columns(df_to_analyze)
df_to_analyze['gender'].value_counts().plot.pie(figsize=(6,8),autopct='%.1f%%')

plt.title('Gender Distribution')

plt.ylabel('')
df_to_analyze['education_level'].value_counts().plot.pie(figsize=(6,8),autopct='%.1f%%')

plt.title('Distribution of educational level',color='black')

plt.ylabel('')
df_to_analyze['age'].value_counts().plot.pie(figsize=(6,8),autopct='%.1f%%')

plt.title('Age Distribution')

plt.ylabel('')
df_to_analyze['job_title'].value_counts().plot.pie(figsize=(6,8),autopct='%.1f%%')

plt.title('Job title Distribution')

plt.ylabel('')
df_to_analyze['company_size'].value_counts().plot.pie(figsize=(6,8),autopct='%.1f%%')

plt.title('company size Distribution')

plt.ylabel('')
df_to_analyze['current_country'].replace({'United States of America':'USA',

                                         'United Kingdom of Great Britain and Northern Ireland':'UK',

                                         'Hong Kong (S.A.R.)':'Hong Kong'},inplace=True)

colors=['red','orange','yellow','green','blue']

df_to_analyze[(df_to_analyze['current_country']=='India') | 

              (df_to_analyze['current_country']=='USA') | 

              (df_to_analyze['current_country']=='Brail') | 

              (df_to_analyze['current_country']=='Japan') | 

              (df_to_analyze['current_country']=='Russia') | 

              (df_to_analyze['current_country']=='China')]['current_country'].value_counts(normalize=True).plot.bar(color=colors)

plt.title('Distribution plot for top 5 countries ')

plt.xlabel('Country')

plt.ylabel('percentage%')
top_5=df_to_analyze[(df_to_analyze['current_country']=='India') | 

              (df_to_analyze['current_country']=='USA') | 

              (df_to_analyze['current_country']=='Brail') | 

              (df_to_analyze['current_country']=='Japan') | 

              (df_to_analyze['current_country']=='Russia') | 

              (df_to_analyze['current_country']=='China')]

top_5_mf=top_5[(top_5['gender']=='Male') | (top_5['gender']=='Female')]

sns.countplot(x='gender',hue='current_country',data=top_5_mf)

plt.title('Top 5 Countries Gender comperison')

plt.legend(loc='upper center')
sns.countplot(x='gender',hue='age',data=top_5_mf)

plt.title('Top 5 Countries Gender,Age  comperison')

plt.legend(loc='upper right')
lst=top_5_mf['education_level'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(20,13))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,3,i+1)

    c=top_5_mf[top_5_mf['education_level']==lst[i]]['gender'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,color=['orange','green'],width=0.4)

    ax.set_title(lst[i])

    ax.set_ylabel('')

plt.show()    
lst=top_5_mf['job_title'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(25,13))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=top_5_mf[top_5_mf['job_title']==lst[i]]['gender'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,color=['red','blue'],width=0.4)

    ax.set_title(lst[i])

    ax.set_ylabel('')

plt.show()    
arab=df_to_analyze[(df_to_analyze['current_country']=='Morocco') | 

              (df_to_analyze['current_country']=='Egypt') | 

              (df_to_analyze['current_country']=='Algeria') | 

              (df_to_analyze['current_country']=='Tunisia') | 

              (df_to_analyze['current_country']=='Saudi Arabia')]

arab_mf=arab[(arab['gender']=='Male') | (arab['gender']=='Female')]

sns.countplot(x='gender',hue='current_country',data=arab_mf)

plt.title('Arab Countries Gender comperison')

plt.legend(loc='upper left')
sns.countplot(x='gender',hue='age',data=arab_mf)

plt.title('Arab Countries Gender,Age  comperison')

plt.legend(loc='upper right')
lst=arab_mf['job_title'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(25,13))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=arab_mf[arab_mf['job_title']==lst[i]]['gender'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,color=['red','blue'],width=0.4)

    ax.set_title(lst[i])

    ax.set_ylabel('')

plt.show()    
lst=arab_mf['job_title'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(30,16))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,6,i+1)

    c=arab_mf[arab_mf['job_title']==lst[i]]['age'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,width=0.4)

    ax.set_title(lst[i])

    ax.set_ylabel('')

plt.show()    
lst=arab_mf['education_level'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(20,13))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,3,i+1)

    c=arab_mf[arab_mf['education_level']==lst[i]]['gender'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,color=['orange','green'],width=0.4)

    ax.set_title(lst[i])

    ax.set_ylabel('')

plt.show()    
lst=arab_mf['education_level'].dropna().unique().tolist()

lst_len=len(lst)

fig=plt.figure(figsize=(20,13))

for i in range(0,lst_len-1):

    ax=fig.add_subplot(2,3,i+1)

    c=arab_mf[arab_mf['education_level']==lst[i]]['job_title'].value_counts()

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,width=0.4)

    ax.set_title(lst[i])

    plt.xticks(rotation=90)

    plt.subplots_adjust(hspace=0.5)

    ax.set_ylabel('')

plt.show()    
arab_DS=arab_mf[arab_mf['job_title']=='Data Scientist']

arab_DS['current_country'].value_counts().plot.bar()

plt.title('number of Data Scientist lives in Arab Countries')

plt.xlabel('Country')

plt.ylabel('number of Data Scientist')