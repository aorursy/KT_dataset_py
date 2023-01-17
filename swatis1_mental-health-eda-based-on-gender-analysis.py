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
mental = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')
mental.head()
mental.tail()
mental.shape
mental.columns
mental[mental==0].count()
mental.isnull().sum()
mental.duplicated().sum()
mental.dtypes
#interested for country specific data

mental= mental.drop(['Timestamp', 'state', 'comments'], axis=1)
cat_columns= mental.loc[:, ['self_employed', 'work_interfere']]
for i in cat_columns:

    mental[i].fillna(mental[i].mode()[0], inplace= True)
#checking whether the null values have been treated

mental.isnull().sum()
mental['Age'].max()
mental['Age'].min()
# since 18 is the minimum age to work officially

def clean_age (age):

    if age>=18 and age<=75:

        return age

    else:

        return np.nan
#created a new column age_clean

mental['age_clean']= mental['Age'].apply(clean_age)
mental['age_clean'].unique()
#treating missing value in the column age_clean

mental['age_clean']= mental['age_clean'].fillna(mental['age_clean'].mean())
mental.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(mental['age_clean'], bins=20)
mental['Gender'].unique()
#will decrease the number of categoried in Gender

mental['Gender']= mental['Gender'].replace({'male':'Male', 'm':'Male', 'Male-ish':'Male', 'maile':'Male', 

                                                'something kinda male?':'Male', 'Cis Male':'Male', 'Mal':'Male', 

                                               'Mal':'Male', 'Male (CIS)':'Male', 'Make':'Male', 

                                                'male leaning androgynous':'Male', 'Man':'Male', 'msle':'Male', 'Mail':'Male', 

                                                'cis male':'Male', 'Malr':'Male', 'Cis Man':'Male', 

                                                'ostensibly male, unsure what that really means':'Male', 'M':'Male',

                                                    'Female':'Female', 'female':'Female', 'Cis Female':'Female',

                                               'F':'Female', 'Woman':'Female', 'f':'Female', 'Femake':'Female', 

                                                'woman':'Female', 'Female ':'Female', 'cis-female/femme':'Female', 

                                               'Female (cis)':'Female', 'femail':'Female', 'Male ':'Male', 

                                                'Trans-female': 'Female(Trans)', 'Trans woman': 'Female(Trans)', 

                                                'Female (trans)': 'Female(Trans)', 'queer/she/they':'Others', 

                                                'non-binary':'Others', 'Nah':'Others', 'All':'Others', 'Enby':'Others', 

                                                'fluid':'Others', 'Genderqueer':'Others', 'Androgyne':'Others', 

                                                'Agender':'Others', 'Guy (-ish) ^_^':'Others', 'Neuter':'Others', 

                                                'queer':'Others', 'A little about you':'Others', 'p':'Others'})
#a huge gap between the male and female workforce in tech companies

sns.countplot(mental['Gender'])
#country wise representation of data with focus on India

plt.figure(figsize=(20,8))

sns.countplot(mental.Country, order= mental['Country'].value_counts().index)

plt.xticks(rotation=90)

plt.annotate('Mental Health Survey Participants from India', xy=(8, 20), xytext=(10, 20.5),

             arrowprops=dict(facecolor='black', shrink=0.05),)
#country-specific data for number of employees ready to seek treatment for their mental health issues

plt.figure(figsize= (18,8))

sns.countplot(x='Country', order= mental['Country'].value_counts().index, hue='treatment', data=mental)

plt.xticks(rotation=90)
#country- wise gender ratio participating in the survey

#shows that more number of males are working in tech companies all over the world

plt.figure(figsize= (18,8))

sns.countplot(x='Country', order= mental['Country'].value_counts().index, hue='Gender', data=mental)

plt.xticks(rotation=90)
#in most of the countries people are generally reluctant to seek treatment for mental health issues

plt.figure(figsize= (18,8))

sns.countplot(x='Country', order= mental['Country'].value_counts().index, hue='treatment', data=mental)

plt.xticks(rotation=90)
#we can see that females are more ready to go for mental health treatment

#but still this cannot be generalized because of huge difference between the number of males and females

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'treatment', data= mental)

plt.title('Seeking Mental Health Treatment based on Gender')

plt.show()
sns.countplot(x= 'treatment', data=mental)
#people who are taking treatment for mental health isuues feels that their work gets affected more

#as compared to the people who are not taking any treatment for their mental health issues

pd.pivot_table(mental, index= 'treatment', columns='work_interfere', values= 'Gender', aggfunc= 'count')
#working remotely does not have any positive effect on mental health issues 

#as males still feels that mental health is affecting tehir work 

pd.pivot_table(mental, index= ['treatment', 'Gender'], columns='work_interfere', 

               values= 'remote_work', aggfunc= 'count')
#males are more reluctant towards getting a treatment for mental health issues with incraese in age

plt.figure(figsize= (18,8))

sns.relplot(x= 'age_clean', y= 'work_interfere', size= 'treatment', col= 'Gender', col_wrap=2, data=mental)
#knowledge of mental health benefits to the employees provided by their employers

#suggest that many people are not aware of any such benefits

plt.figure(figsize=(18,8))

sns.countplot(x= 'tech_company', hue= 'benefits', data= mental)
#discussion of mental health in employee wellness program by their employers

#suggest that employers have not even discussed about mental health in their wellness program with their employees

plt.figure(figsize=(18,8))

sns.countplot(x= 'tech_company', hue= 'wellness_program', data= mental)
#assistance pprovided by employers for any help related to mental health issues

#either people are not aware or no resources are provided to learn about mental health issues and seek help by the employers

plt.figure(figsize=(18,8))

sns.countplot(x= 'tech_company', hue= 'seek_help', data= mental)
#protection of anonymity by the employers if seeking any help

plt.figure(figsize=(18,8))

sns.countplot(x= 'tech_company', hue= 'anonymity', data= mental)
#able to take leave for a mental health issue

plt.figure(figsize=(18,8))

sns.countplot(x= 'tech_company', hue= 'leave', data= mental)
#discussion with coworkers regarding mental health issues

#suggest that both the genders may discuss their issues with their coworkers

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'coworkers', data= mental)
#discussion with supervisor for mental health issues

#males are more likely to discuss their mental health issues with their supervisor as compared to their female counterparts

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'supervisor', data= mental)
#no one wants to discuss their mental health issues with their prospective employers

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'mental_health_interview', data= mental)
#males are more likely to discuss their physical health issues with their prospective employers as compared to females

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'phys_health_interview', data= mental)
#employees are not aware whether their mental health issues are taken as seriously 

#as their physical health issues by their employers

plt.figure(figsize= (18,8))

sns.countplot(x= 'Gender', hue= 'mental_vs_physical', data= mental)