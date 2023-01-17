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
#questions 

questions=pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

questions.T
##questions and multiple choice responses

mcr=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

mcr
mcr['Q11'].unique()
#describe the most common answer of each question

mcr.describe()
import matplotlib.pyplot as plt

nationality=mcr['Q3'].value_counts()

print(nationality)
numbers=list(nationality)
import matplotlib.pyplot as plt



import seaborn as sns

##mcr.assign(dummy = 1).groupby(

  ##['dummy','Q3']

##).size().to_frame().unstack().plot(kind='bar',stacked=True,legend=False)



##plt.title('Country distribution')



# other it'll show up as 'dummy' 

##plt.xlabel('country')



# disable ticks in the x axis

#plt.xticks([])



# fix the legend

#current_handles, _ = plt.gca().get_legend_handles_labels()

#reversed_handles = reversed(current_handles)



#labels = reversed(mcr['Q3'].unique())



#plt.legend(reversed_handles,labels,loc='lower right')

#plt.show()
plt.figure(figsize=(20,60))

ax=mcr['Q3'].value_counts()[:59].plot.barh(width=0.6,color=sns.color_palette('Set2',25))

plt.gca().invert_yaxis()

plt.title('Country')

plt.show()
import seaborn as sns

age = mcr['Q1'][1:].dropna()

d= ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']

plt.figure(figsize=(12,6))

sns.countplot(age,order=d)

plt.xlabel('Age Groups')

plt.ylabel('Count')

plt.title('Age Groups Distribution')

plt.show()
mcr['Q2'].unique()

sex = mcr['Q2'][1:].dropna()

d= [ 'Male', 'Female',

       'Prefer to self-describe', 'Prefer not to say']

plt.figure(figsize=(12,6))

sns.countplot(sex,order=d)

plt.xlabel('Sex')

plt.ylabel('Count')

plt.title('Sex s Distribution')

plt.show()
mcr['Q4'].unique()

education = mcr['Q4'][1:].dropna()

d= [ 'Master’s degree', 'Professional degree', 'Bachelor’s degree',

       'Some college/university study without earning a bachelor’s degree',

       'Doctoral degree', 'I prefer not to answer',

       'No formal education past high school']

plt.figure(figsize=(18,6))

sns.countplot(education,order=d)

plt.xlabel('Education')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Education Distribution')

plt.show()
# Most common roles of the survey

plt.figure(figsize=(10,10))

ax=mcr['Q5'].value_counts()[:12].plot.barh(width=0.7,color=sns.color_palette('Set2',25))

plt.gca().invert_yaxis()

plt.title('Job Roles 2019')

plt.show()
plt.figure(figsize=(10,10))

ax=mcr['Q10'].value_counts()[:15].plot.barh(width=0.7,color=sns.color_palette('Set3',25))

plt.gca().invert_yaxis()

plt.title('yearly compensation')

plt.show()
plt.figure(figsize=(10,10))

ax=mcr['Q19'].value_counts()[:10].plot.barh(width=0.8,color=sns.color_palette('Set1',25))

plt.gca().invert_yaxis()

plt.title('Programming Language used')

plt.show()
mcr['Q6'].unique()

sex = mcr['Q6'][1:].dropna()

d= [ '1000-9,999 employees', '> 10,000 employees', 

       '0-49 employees', '50-249 employees', '250-999 employees']

plt.figure(figsize=(12,6))

sns.countplot(sex,order=d)

plt.xlabel('No. of Employees')

plt.ylabel('Count')

plt.title('Size of the Company')

plt.show()
#Top five tools used to analyze data

plt.figure(figsize=(10,6))

ax=mcr['Q14'].value_counts()[:5].plot.barh(width=0.9,color=sns.color_palette('Set2',15))

plt.gca().invert_yaxis()

plt.title('primary tool used to analyze data')

plt.show()
#Years writing code to analyze code

plt.figure(figsize=(10,10))

ax=mcr['Q15'].value_counts()[:7].plot.barh(width=0.8,color=sns.color_palette('Set1',15))

plt.gca().invert_yaxis()

plt.title('Years writing code')

plt.show()
#People using TPU

plt.figure(figsize=(10,10))

ax=mcr['Q22'].value_counts()[:5].plot.barh(width=0.7,color=sns.color_palette('Set2',15))

plt.gca().invert_yaxis()

plt.title('TPU used')

plt.show()
# Years using machine learning methods

plt.figure(figsize=(10,10))

ax=mcr['Q23'].value_counts()[:7].plot.barh(width=0.9,color=sns.color_palette('Set2',15))

plt.gca().invert_yaxis()

plt.title('Years using machine learning methods')

plt.show()
#Years writing code to analyze code

plt.figure(figsize=(10,10))

ax=mcr['Q11'].value_counts()[:6].plot.barh(width=0.8,color=sns.color_palette('Set1',15))

plt.gca().invert_yaxis()

plt.title('spent on machine learning and/or cloud computing products at work in past 5 years')

plt.show()
mcr.groupby(['Q3','Q19']).size().unstack().plot(kind='bar',stacked=True,figsize =(50,20))

plt.show()
mcr.groupby(['Q2','Q4']).size().unstack()[:4].plot(kind='bar',stacked=True,figsize =(20,10))

plt.show()
mcr.groupby(['Q2','Q1']).size().unstack()[:4].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Gender distribution')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Age group enganged in both male and female')

plt.show()
mcr.groupby(['Q4','Q5']).size().unstack()[:7].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Education distribution')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Education Distribution and their job roles')

plt.show()
mcr.groupby(['Q5','Q10']).size().unstack()[:7].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Job distribution')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Job Distribution and their salaries')

plt.show()
mcr.groupby(['Q6','Q23']).size().unstack()[:5].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Company size')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Company size and their employee experience using machine learning methods')

plt.show()
mcr.groupby(['Q5','Q22']).size().unstack()[:9].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Job Roles')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('TPU used by various professionals')

plt.show()
mcr.groupby(['Q2','Q15']).size().unstack()[:4].plot(kind='bar',stacked=True,figsize =(20,10))

plt.xlabel('Gender')

plt.ylabel('Count')

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.title('Years of experience to code according to Gender')

plt.show()
survey=pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

survey.T
otr=pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

otr