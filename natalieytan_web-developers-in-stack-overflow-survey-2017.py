import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
surveyresults = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent')



print (surveyresults.shape, surveyresults.columns)
surveyresults['WebDeveloperType'].unique()
surveyresults['WebDeveloperType'].value_counts()
%matplotlib inline



webdevcount = pd.value_counts(surveyresults['WebDeveloperType'].values, sort=True)



webdevcount.plot(kind='bar', title='Web Developer Types')

plt.ylabel('Count')

plt.show()



surveyresults['WebDeveloperType'].notnull().sum()
surveyresults[['WebDeveloperType','HaveWorkedLanguage', 'HaveWorkedFramework', 'HaveWorkedDatabase', 'HaveWorkedPlatform', 'YearsCodedJob', 'YearsProgram', 'FormalEducation', 'Gender', 'MajorUndergrad', 'Salary']][surveyresults.WebDeveloperType.notnull()].isnull().sum().sort_values(ascending=False)
webdeveloperlanguages = surveyresults[['WebDeveloperType','HaveWorkedLanguage']].dropna(axis=0, how='any')



webdeveloperlanguages['NumberofLanguages'] = webdeveloperlanguages['HaveWorkedLanguage'].apply(lambda x: len(str(x).split(';')))



webdeveloperlanguages.head()
webdeveloperlanguages.groupby('WebDeveloperType')['NumberofLanguages'].describe()

plt.hist(webdeveloperlanguages['NumberofLanguages'], bins=range(1,18), normed=True)

plt.title('Number of Langauges Web Developers (All Types) have worked with in Past Year')

plt.ylabel('Proportion of Web Developers')

plt.xlabel('Number of Languages')

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))



webdeveloperlanguages['NumberofLanguages'].hist(by=webdeveloperlanguages['WebDeveloperType'], bins=range(1,18), normed=True, ax=axes)

plt.suptitle('Number of Langauges Web Developers have worked with in Past Year', x=0.5, y=1.05, ha='center', fontsize='xx-large')

fig.text(0.5, 0.0, 'Number of Languages', ha='center')

fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')



plt.tight_layout()

plt.show()

sns.boxplot(x='WebDeveloperType', y='NumberofLanguages', data=webdeveloperlanguages)

plt.title('Number of Languages Web Developers have Worked with in Past Year')

plt.xlabel('Web Developer Type')

plt.ylabel('Number of Languages')

plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)

plt.show()
def webDevExpandedTable(column):

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    expanded = df[column].str.replace(' ', '').str.get_dummies(';')

    fulldf = pd.concat([df['WebDeveloperType'], expanded], axis=1).groupby('WebDeveloperType').sum().T.rename_axis(column)

    return fulldf
def webDevExpandedProportionTable(column):

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    counts = {'Back-end Web developer': df.WebDeveloperType.value_counts()['Back-end Web developer'],

              'Front-end Web developer': df.WebDeveloperType.value_counts()['Front-end Web developer'],

              'Full stack Web developer': df.WebDeveloperType.value_counts()['Full stack Web developer']

             }  

    

    

    expanded = df[column].str.replace(' ', '').str.get_dummies(';')

    fulldf = pd.concat([df['WebDeveloperType'], expanded], axis=1).groupby('WebDeveloperType').sum().T.rename_axis(column)

   

    for column in list(fulldf.columns):

        fulldf[column] =fulldf[column].apply(lambda x: round((x/counts[column]), 3))

    

    return fulldf
def top(df, number):

    top = {}

    rank = np.arange(1, number+1)

    for column in list(df.columns):

        topdf = df[column].sort_values(ascending=False).head(number)

        series = []

        for index, row in topdf.iteritems():

            series.append(str(index) + ' ( ' + str(row) + ' ) ')

        top[column] = series

        

        

    return pd.DataFrame(top, index=rank)
webDevExpandedTable('HaveWorkedLanguage')
languages = webDevExpandedProportionTable('HaveWorkedLanguage')

languages
top(languages, 10)
selected = list(languages.mean(axis=1).sort_values(ascending=False).head(10).index)



languages.loc[selected].plot(kind='bar')

plt.title('Languages Web Developers Have Worked With in Past Year')

plt.xlabel('Programming Language')

plt.ylabel('Proportion')

plt.legend(title='Web Developer Type')
webDevExpandedTable('HaveWorkedFramework')
frameworks = webDevExpandedProportionTable('HaveWorkedFramework')

frameworks
top(frameworks, 5)
webDevExpandedTable('HaveWorkedDatabase')
database = webDevExpandedProportionTable('HaveWorkedDatabase')

database
top(database, 5)
topdatabase = list(database.mean(axis=1).sort_values(ascending=False).head(5).index)



database.loc[topdatabase].plot(kind='bar')

plt.title('Databases Web Developers Have Worked With in Past Year')

plt.xlabel('Database')

plt.ylabel('Proportion')

plt.legend(title='Web Developer Type')
webDevExpandedTable('HaveWorkedPlatform')
platforms = webDevExpandedProportionTable('HaveWorkedPlatform')

platforms
top(platforms, 5)
ide = webDevExpandedProportionTable('IDE')

ide
top(ide, 10)
topide = list(ide.mean(axis=1).sort_values(ascending=False).head(10).index)



ide.loc[topide].plot(kind='bar')

plt.title('IDEs Developers Use')

plt.xlabel('IDE')

plt.ylabel('Proportion')

plt.legend(title='Web Developer Type')
def mapyears(column):

    yeardict = {

    'Less than a year': 0.5,

    '1 to 2 years': 1.5,

    '2 to 3 years': 2.5,

    '3 to 4 years': 3.5,

    '4 to 5 years': 4.5,

    '6 to 7 years': 6.5, 

    '7 to 8 years': 7.5,

    '8 to 9 years': 8.5,

    '9 to 10 years': 9.5,

    '10 to 11 years': 10.5,

    '11 to 12 years': 11.5,

    '12 to 13 years': 12.5,

    '13 to 14 years': 13.5,

    '14 to 15 years': 14.5,

    '15 to 16 years': 15.5,

    '16 to 17 years': 16.5,

    '17 to 18 years': 17.5,

    '18 to 19 years': 18.5,

    '19 to 20 years': 19.5,

    '20 or more years': 20.5

    }

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    df[column] = df[column].map(yeardict)

    return df
webDevYearsCode = mapyears('YearsCodedJob')



webDevYearsCode.groupby('WebDeveloperType').describe()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))



webDevYearsCode.hist(by=webdeveloperlanguages['WebDeveloperType'], bins=8, normed=True, ax=axes)

plt.suptitle('Number of Years Web Developers have Coded on Jobs for', x=0.5, y=1.05, ha='center', fontsize='xx-large')

fig.text(0.5, 0.0, 'Number of Years', ha='center')

fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')



plt.tight_layout()

plt.show()

sns.boxplot(x='WebDeveloperType', y='YearsCodedJob', data=webDevYearsCode)

plt.title('Number of Years Web Developers have Coded on Jobs for')

plt.xlabel('Web Developer Type')

plt.ylabel('Number of Years')

plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)

plt.show()
webDevYearsProgram = mapyears('YearsProgram')



webDevYearsProgram.groupby('WebDeveloperType').describe()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))



webDevYearsProgram.hist(by=webdeveloperlanguages['WebDeveloperType'], bins=8, normed=True, ax=axes)

plt.suptitle('Number of Years Web Developers have Coded for', x=0.5, y=1.05, ha='center', fontsize='xx-large')

fig.text(0.5, 0.0, 'Number of Years', ha='center')

fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')



plt.tight_layout()

plt.show()
sns.boxplot(x='WebDeveloperType', y='YearsProgram', data=webDevYearsProgram)

plt.title('Number of Years Web Developers have Coded for')

plt.xlabel('Web Developer Type')

plt.ylabel('Number of Years')

plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)

plt.show()
def crossCategory(column):

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)

    return cross.T
def crossProp(column):

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)

    return cross.div(cross['All'],axis=0).round(3).T.sort_values(by='All', ascending=False).drop('All', 0)
def crossCategorybyCat(column):

    df = surveyresults[['WebDeveloperType', column]].dropna(axis=0, how='any')

    cross = pd.crosstab(index=df['WebDeveloperType'],  columns=df[column], margins=True)

    cross = cross/cross.loc['All']

    return cross.T
crossCategory('FormalEducation')
crossProp('FormalEducation')
education = crossProp('FormalEducation').drop('All', 1)

education.rename(columns={}, inplace=True)



education.plot(kind='bar')

plt.title('Highest Level of Formal Education Web Developers Have Achieved')

plt.ylabel('Proportion')

plt.xlabel('Education Level')

edlabels = list(education.index)

edlabels[2] = 'College/University without Degree'

plt.xticks(np.arange(0,9), edlabels)

plt.legend(title='Web Developer Type')
crossCategory('MajorUndergrad')
crossProp('MajorUndergrad')
undergrad = crossProp('MajorUndergrad').drop('All', 1).head(10)



undergrad.plot(kind='bar')

plt.title('Undergraduate Degrees attained by Web Developers')

plt.ylabel('Proportion')

plt.xlabel('Undergraduate Degree')

plt.legend(title='Web Developer Type')
pd.DataFrame(crossProp('MajorUndergrad').loc['Computer science or software engineering':'Information technology, networking, or system administration', :].sum())
webDevSatisfaction = surveyresults[['WebDeveloperType','CareerSatisfaction']].dropna(axis=0, how='any')



webDevSatisfaction.groupby('WebDeveloperType').describe()
sns.boxplot(x='WebDeveloperType', y='CareerSatisfaction', data=webDevSatisfaction)

plt.title('Web Developer Career Satisfcation')

plt.xlabel('Web Developer Type')

plt.ylabel('Career Satisfaction')

plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))



webDevSatisfaction.hist(by=webDevSatisfaction['WebDeveloperType'], bins=range(0,10), normed=True, ax=axes)

plt.suptitle('Career Satisfaction of Web Developers', x=0.5, y=1.05, ha='center', fontsize='xx-large')

fig.text(0.5, 0.0, 'Career Satisfaction', ha='center')

fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')



plt.tight_layout()

plt.show()

webDevSalary = surveyresults[['WebDeveloperType','Salary']][surveyresults['Salary'] > 0].dropna(axis=0, how='any')



webDevSalary.groupby('WebDeveloperType').describe()
print(len(webDevSalary[webDevSalary.Salary < 1]), len(webDevSalary[webDevSalary.Salary < 1000]), len(webDevSalary))
sns.boxplot(x='WebDeveloperType', y='Salary', data=webDevSalary)

plt.title('Web Developer Salary')

plt.xlabel('Web Developer Type')

plt.ylabel('Salary')

plt.xticks([0, 1, 2], ['Full stack', 'Back-end', 'Front-end'], rotation=40)

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 5))



webDevSalary.hist(by=webDevSalary['WebDeveloperType'], bins=10, normed=True, ax=axes)

plt.suptitle('Web Developer Salary', x=0.5, y=1.05, ha='center', fontsize='xx-large')

fig.text(0.5, 0.0, 'Salary', ha='center')

fig.text(0.0, 0.5, 'Proportion of Web Developers', va='center', rotation='vertical')



plt.tight_layout()

plt.show()

multidata = surveyresults[['WebDeveloperType','Salary', 'YearsCodedJob']].dropna(axis=0, how='any')



multidata = multidata[multidata.Salary > 0]



multidata['YearsCodedJob'] = multidata['YearsCodedJob'].map({

    'Less than a year': 0.5,

    '1 to 2 years': 1.5,

    '2 to 3 years': 2.5,

    '3 to 4 years': 3.5,

    '4 to 5 years': 4.5,

    '6 to 7 years': 6.5, 

    '7 to 8 years': 7.5,

    '8 to 9 years': 8.5,

    '9 to 10 years': 9.5,

    '10 to 11 years': 10.5,

    '11 to 12 years': 11.5,

    '12 to 13 years': 12.5,

    '13 to 14 years': 13.5,

    '14 to 15 years': 14.5,

    '15 to 16 years': 15.5,

    '16 to 17 years': 16.5,

    '17 to 18 years': 17.5,

    '18 to 19 years': 18.5,

    '19 to 20 years': 19.5,

    '20 or more years': 20

    })





sns.lmplot(y='Salary', x='YearsCodedJob', data=multidata, hue='WebDeveloperType', x_jitter=0.7, fit_reg=False)

plt.xlabel('Number of Years Developer has Coded for Job')

plt.title('Relationship between Salary and Years of Coding for Job')
crossProp('Gender')
webDevByGender = surveyresults[['WebDeveloperType', 'Gender']][pd.notnull(surveyresults['WebDeveloperType'])]



gender = ['Male', 'Female', 'Other', 'Transgender', 'Gender non-conforming', np.NaN]



webDevByGender['Gender'] = webDevByGender['Gender'].apply(lambda i: i if i in gender else 'Multiple')





webDevGenderFullTab = pd.crosstab(index=webDevByGender['WebDeveloperType'],  columns=webDevByGender['Gender'], margins=True)



webDevGenderFullTab.T.sort_values(by='All', ascending=False)
allgender = webDevGenderFullTab.div(webDevGenderFullTab["All"],axis=0).T.sort_values(by='All', ascending=False).drop('All', 0)

allgender
allgenders = allgender.drop('All', 1)



allgenders.plot(kind='bar')

plt.title('Gender Ratio by Web Developer Type')

plt.ylabel('Proportion')

plt.xlabel('Gender')

plt.legend(title='Web Developer Type')
webDevMaleFemale = webDevByGender[(webDevByGender.Gender=='Male') | (webDevByGender.Gender=='Female')]



webDevMaleFemaleCrossTab = pd.crosstab(index=webDevMaleFemale['WebDeveloperType'],  columns=webDevMaleFemale['Gender'], margins=True)



webDevMaleFemaleCrossTab
webDevMaleFemaleProp = webDevMaleFemaleCrossTab.div(webDevMaleFemaleCrossTab["All"],axis=0)

webDevMaleFemaleProp 
gender = webDevMaleFemaleProp.sort_values(by='All', ascending=False).drop('All', 0).drop('All', 1)



gender.plot(kind='bar')

plt.title('Male / Female Ratio by Web Developer Type')

plt.ylabel('Proportion')

plt.xlabel('Web Developer Type')

plt.legend(title='Web Developer Type')
webDevMaleFemaleCrossTab/webDevMaleFemaleCrossTab.loc['All']
genderDivison = webDevMaleFemaleCrossTab/webDevMaleFemaleCrossTab.loc['All']



genderDivison.drop('All', 0).T.plot(kind='bar')

plt.title('Proportion of Web Developer Type by Gender')

plt.ylabel('Proportion')

plt.legend(title='Web Developer Type')