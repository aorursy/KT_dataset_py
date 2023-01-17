#Data Scientist is the sexiest job of the 21st century, who want to be a data scientist? 

#Maybe we can get answers from this survey.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_digits



# Load the dataset 

df_mcr= pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

df_cr= pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1", low_memory=False)

df_ffr= pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)



# creat two dataset wiht careerswitcher variable

df_mcr_cs=df_mcr[df_mcr['CareerSwitcher']=='Yes']

df_mcr_ns=df_mcr[df_mcr['CareerSwitcher']=='No']
#based on Gender Select, there are much more men than women to ansewer the survey.

#But the proportion of women who want to change job is higher than men.  

sns.countplot(y="GenderSelect",hue="CareerSwitcher", data=df_mcr,order=df_mcr_cs['GenderSelect'].value_counts().index, palette="Set2");
# Top three countries: India, US and Russia

# there are more indian persons who want to change job

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(12,12)

sns.countplot(y="Country", hue="CareerSwitcher",data=df_mcr, order=df_mcr_cs['Country'].value_counts().index, palette="Set2");

sns.despine()
sns.countplot(y="EmploymentStatus", hue="CareerSwitcher",data=df_mcr, order=df_mcr_cs['EmploymentStatus'].value_counts().index, palette="Set2");
# more younger people(20-30) want to change job compared to 40-50 years old persons

df_mcr['Age'].dropna()

g = sns.FacetGrid(df_mcr, hue="CareerSwitcher", size=4, aspect=1.3)

g.map(plt.hist, "Age", histtype='barstacked', stacked=True);

plt.legend();
#Which Machine learning tool the Career Switcher prefer to learn? 

#TensorFlow is on the top of the list, then the second one is Python.

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(12,14)

sns.countplot(y="MLToolNextYearSelect",data=df_mcr_cs,order=df_mcr_cs['MLToolNextYearSelect'].value_counts().index, palette="Set2");

sns.despine()
#Which Machine learning method the Career Switcher prefer to learn? 

#Deep Learning is on the top of the list, then the second one is Neural nets.

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(12,14)

sns.countplot(y="MLMethodNextYearSelect",data=df_mcr_cs,order=df_mcr_cs['MLMethodNextYearSelect'].value_counts().index, palette="Set2");

sns.despine()
#most Career Switcher spend between 2-10 hours on learning 

sns.countplot(y="TimeSpentStudying",data=df_mcr_cs,order=df_mcr_cs['TimeSpentStudying'].value_counts().index, palette="Set2");
#most Career Switcher have Bachelor's and Master's degree

sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(12,14)

sns.countplot(y="FormalEducation",data=df_mcr_cs,order=df_mcr_cs['FormalEducation'].value_counts().index, palette="Set2");

sns.despine()
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(12,14)

sns.countplot(y="MajorSelect",data=df_mcr_cs,order=df_mcr_cs['MajorSelect'].value_counts().index, palette="Set2");

sns.despine()
#decision tree to predict who want to change job?

dataset = load_digits()

X, y = dataset.data, dataset.target



columns_to_keep = ['GenderSelect','Country','Age','EmploymentStatus','TimeSpentStudying','FormalEducation','MajorSelect']

df_mcr=df_mcr.fillna('None')  

df_mcr=df_mcr[(df_mcr['CareerSwitcher']!='None')]



df_mcr_data= df_mcr[columns_to_keep]

df_mcr_data=pd.get_dummies(df_mcr_data)

df_mcr_target=df_mcr['CareerSwitcher']

df_mcr_target=pd.get_dummies(df_mcr_target)

X_train, X_test, y_train, y_test = train_test_split(df_mcr_data, df_mcr_target , random_state = 5)



clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

    .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

    .format(clf.score(X_test, y_test))); 

# the result is very good, but too good to be true. May be because of "age" which is a very granular variable. 