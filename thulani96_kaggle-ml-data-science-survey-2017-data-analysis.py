import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
%matplotlib inline
rcParams['figure.figsize'] = 20,8
data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding = "ISO-8859-1")
print(data.shape)

# fig, ax = plt.subplots(2,1, figsize = (20,15))
ax= sns.countplot(x = 'GenderSelect', data = data, palette = 'Set1')
ax.set(title = 'Gender Count as per compreensive view', ylabel = 'Total', xlabel = 'Gender')
plt.show()
fig, ax = plt.subplots(1,1, figsize = (12,20))
sns.countplot(y = 'Country', data = data, palette = 'Set1')
ax.set(title = 'Countries and Number of User who took the survey', ylabel = 'Country', xlabel = 'Total Users')
plt.show()
fig, ax = plt.subplots(1,1, figsize = (10,40))
sns.countplot(y = 'Country', data = data, hue = 'GenderSelect', palette = 'Set1')
ax.set(title = 'Countries and Number of Users who took the survey by Gender', ylabel = 'Country', xlabel = 'Total Users')
plt.show()

fig, ax = plt.subplots(1,1, figsize = (10,10))
sns.countplot(y = 'EmploymentStatus', data = data, palette = 'Set1')
ax.set(title = 'User Employment Status', ylabel = 'Employment Status', xlabel = 'Total Users')
plt.show()
fig, ax = plt.subplots(1,1, figsize = (10,12))
sns.countplot(y = 'EmploymentStatus', data = data,hue='GenderSelect', palette = 'Set1')
ax.set(title = 'User Employment Status', ylabel = 'Employment Status', xlabel = 'Total Users')
plt.show()
# First of all since we have NaNs in the Career Switcher column, we have to assume the user decided
# Not to specify, thus we replace the 'NaN' object with the text 'Unspecified' to make our job easier to do.

data['CareerSwitcher'].fillna('0', inplace = True)
data['CareerSwitcher'] = data['CareerSwitcher'].map({
    '0': 'Unspecified',
    'Yes':'Yes',
    'No':'No'
})
fig, ax = plt.subplots(1,2,figsize = (15,6))
sns.countplot(x = 'CareerSwitcher', data = data, palette = 'Set1',ax=ax[0])
sns.countplot(x = 'CareerSwitcher', hue = 'GenderSelect', palette = 'Set1', data = data,ax=ax[1])
ax[0].set(title = 'Career Switcher Distribution', ylabel= 'Total', xlabel = 'Career Switcher')
ax[1].set(title = 'Career Switcher Distribution By Gender', ylabel= 'Total', xlabel = 'Career Switcher')
plt.show()
data['Age'].fillna(0, inplace = True)
fig, ax = plt.subplots(2,1, figsize = (15,9))
sns.factorplot(x = 'GenderSelect', y = 'Age',ax = ax[0], data = data, kind = 'violin')
sns.distplot(data['Age'],ax = ax[1])
ax[0].set(title = 'Gender x Age', xlabel = 'Gender')
ax[1].set(title = 'Distribution Plot of Age')
plt.show()
data['CurrentJobTitleSelect'].fillna('Unspecified', inplace = True)
data['CurrentJobTitleSelect'].unique()
fig, ax = plt.subplots(1,1, figsize = (10, 10))
sns.countplot(y = 'CurrentJobTitleSelect', data = data, palette = 'Set1')
ax.set(title = 'Job Titles Selection', ylabel = 'Job Title', xlabel = 'Total people')

fig, ax = plt.subplots(1,1, figsize = (10, 10))
sns.countplot(y = 'CurrentJobTitleSelect', hue = 'GenderSelect', data = data, palette = 'Set1')
ax.set(title = 'Job Title selections By Gender', ylabel = 'Job Title', xlabel = 'Total people (By gender)')



plt.show()
data['CurrentJobTitleSelect'].unique()
# LETS SHORTEN THE JOB TITLES FOR A CLEANER PLOT
data['CurrentJobTitleSelect'] = data['CurrentJobTitleSelect'].map({
    'DBA/Database Engineer' : 'DBA',
    'Unspecified': 'X',
    'Operations Research Practitioner': 'ORP',
    'Computer Scientist': 'CS',
    'Data Scientist': 'DS',
    'Software Developer/Software Engineer':'SE',
    'Business Analyst': 'BA',
    'Engineer':'EN',
    'Scientist/Researcher': 'S/R',
    'Researcer':'RES',
    'Other':'O',
    'Data Analyst': 'DA',
    'Machine Learning Engineer': 'MLE',
    'Statistician': 'ST',
    'Predictive Modeler': 'PM',
    'Programmer':'PR',
    'Data Miner':'DM'
})
data['LearningDataScience'].fillna('Unspecified', inplace = True)
data['LearningDataScience'].unique()
fig, ax = plt.subplots(1,1, figsize = (10, 5))
sns.countplot(x = 'CurrentJobTitleSelect',data = data, hue='LearningDataScience')
ax.set(title = 'Job Title selections By Gender', ylabel = 'Total', xlabel = 'Job Titles as categorized by Gender')
plt.show()
data['CodeWriter'].fillna('Unspecified', inplace = True)
# data['CodeWriter']
fig, ax = plt.subplots(nrows = 3, figsize = (15,15))
sns.countplot(x = 'CodeWriter', data = data, hue = 'GenderSelect',ax = ax[0])
sns.countplot(x = 'CodeWriter', data = data, palette = 'Set1', hue = 'CurrentJobTitleSelect',ax = ax[1])
sns.countplot(x = 'CodeWriter', data = data, ax = ax[2])
ax[0].set(title = 'Code Writer or Not')
ax[1].set(title = 'Code writer By Job')
ax[2].set(title = 'Code Writer or Not In Total')
data['StudentStatus'].unique()