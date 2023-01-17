import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.set(style="darkgrid")

data = pd.read_csv("/kaggle/input/mental-health-in-tech-survey/survey.csv")

# description of each column

description = {

"state": "If you live in the United States, which state or territory do you live in?",

"self_employed": "Are you self-employed?",

"family_history": "Do you have a family history of mental illness?",

"work_interfere": "If you have a mental health condition, do you feel that it interferes with your work?",

"no_employees": "How many employees does your company or organization have?",

"remote_work": "Do you work remotely (outside of an office) at least 50% of the time?",

"tech_company": "Is your employer primarily a tech company/organization?",

"benefits": "Does your employer provide mental health benefits?",

"care_options": "Do you know the options for mental health care your employer provides?",

"wellness_program": "Has your employer ever discussed mental health as part of an employee wellness program?",

"seek_help": "Does your employer provide resources to learn more about mental health issues and how to seek help?",

"anonymity": "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",

"leave": "How easy is it for you to take medical leave for a mental health condition?",

"mental_health_consequence": "Do you think that discussing a mental health issue with your employer would have negative consequences?",

"phys_health_consequence": "Do you think that discussing a physical health issue with your employer would have negative consequences?",

"coworkers": "Would you be willing to discuss a mental health issue with your coworkers?",

"supervisor": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",

"mental_health_interview": "Would you bring up a mental health issue with a potential employer in an interview?",

"phys_health_interview": "Would you bring up a physical health issue with a potential employer in an interview?",

"mental_vs_physical": "Do you feel that your employer takes mental health as seriously as physical health?",

"obs_consequence": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",

"comments": "Any additional notes or comments",

"treatment": "Have you sought treatment for a mental health condition?",

}
print(f"Min age: {data['Age'].min()}, max age: {data['Age'].max()}, avg age: {data['Age'].mean()}.\n\nUnique genders: \n{data['Gender'].unique()}")
# Data cleaning

data['Age'] = data['Age'][(data['Age'] > 18) & (data['Age'] < 100)]

data = data.dropna(subset=['Age'], axis=0)

data['Age'] = data['Age'].astype('int32')

data['Age_cuts'] = pd.qcut(data['Age'], 6)

data['Gender'] = data['Gender'].replace(['Female', 'female', 'F', 'f', 'Woman', 'woman', 'Female ', 'Female (trans)', 'Female (cis)'], 'Woman')

data['Gender'] = data['Gender'].replace(['Male', 'male', 'M', 'm', 'Man', 'man', 'Male ', 'Make', 'Cis Male', 'Male (CIS)'], 'Man')

data['Gender'] = data['Gender'][(data['Gender'] == 'Woman') | (data['Gender'] == 'Man')]

data['no_employees'] = data['no_employees'].replace('More than 1000', 'Over 1000')

data['no_employees'] = data['no_employees'].astype(pd.api.types.CategoricalDtype(categories=['1-5','6-25','26-100', '100-500', '500-1000', 'Over 1000'])) # for sort purposes

data = data.drop(['Timestamp', 'comments'], axis = 1) # we won't be using those
print(f"Min age: {data['Age'].min()}, max age: {data['Age'].max()}, avg age: {data['Age'].mean()}.\n\nUnique genders: \n{data['Gender'].unique()}")
data.info()
for column in data:

    if data[column].unique().shape[0] < 7:

        print("Unique values for " + column + ": " + str(data[column].unique()))
columns_for_hist = ['Age', 'Gender', 'self_employed', 'family_history', 'phys_health_consequence', 'no_employees', 'work_interfere', 'remote_work', 'tech_company', 'anonymity',

 'care_options', 'wellness_program', 'seek_help',  'leave', 'mental_health_consequence','benefits', 'treatment',

 'mental_vs_physical', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'coworkers', 'obs_consequence', 'Age_cuts']

sns.set_context("paper", rc={"font.size":12,"axes.titlesize":16,"axes.labelsize":20}) 

fig = plt.figure()

fig.set_size_inches(25, 70.5)

fig.subplots_adjust(hspace=0.4, wspace=0.15)

for i in range(1, len(columns_for_hist)+1):

    ax = fig.add_subplot(12, 2, i)

    if columns_for_hist[i-1] in description:

        ax.title.set_text(description[columns_for_hist[i-1]])

    if columns_for_hist[i-1] == 'Age':

        sns.distplot(a=data['Age'], kde=False)

    else:

        sns.countplot(data.loc[:, columns_for_hist[i-1]])
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":16}) 

ax = sns.catplot(x='treatment', col='Age_cuts', kind='count', data=data, col_wrap=3, height=4, aspect=1)

_ = ax.fig.suptitle('treatment' + ': ' + description['treatment'], y=1.05, fontsize = 16)
ax = sns.catplot(x='seek_help', col='Age_cuts', kind='count', data=data, col_wrap=3, height=4, aspect=1)

ax.fig.suptitle('seek_help' + ': ' + description['seek_help'] + "\n\nbenefits: " + description['benefits'], y=1.1, fontsize = 16)

ax = sns.catplot(x='benefits', col='Age_cuts', kind='count', data=data, col_wrap=3, height=4, aspect=1)
ax = sns.catplot(x='no_employees', col='Age_cuts', kind='count', data=data, col_wrap=3, height=4.2, aspect=1.3)
ax = sns.catplot(x='seek_help', col='no_employees', kind='count', data=data, col_wrap=3, height=4, aspect=1)

ax.fig.suptitle('seek_help' + ': ' + description['seek_help'] + "\n\nbenefits: " + description['benefits'], y=1.1, fontsize = 16)

ax = sns.catplot(x='benefits', col='no_employees', kind='count', data=data, col_wrap=3, height=4, aspect=1)
countries = data['Country'].value_counts().index[:4] # 4 countries with biggest size

data_countries = data[data['Country'].isin(countries)]
ax = sns.catplot(x='seek_help', col='Country', kind='count', data=data_countries, col_wrap=4, height=4, aspect=1)

ax.fig.suptitle('seek_help' + ': ' + description['seek_help'] + "\n\nbenefits: " + description['benefits'], y=1.2, fontsize = 16)

ax = sns.catplot(x='benefits', col='Country', kind='count', data=data_countries, col_wrap=4, height=4, aspect=1)