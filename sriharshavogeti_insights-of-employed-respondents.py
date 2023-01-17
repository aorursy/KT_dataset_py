# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



dataset = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')

columns_needed = ['Country','GenderSelect','Age','MLToolNextYearSelect','CurrentJobTitleSelect','LanguageRecommendationSelect','LearningPlatformSelect','JobSkillImportanceDegree','JobSkillImportanceKaggleRanking','JobSkillImportanceMOOC','FormalEducation','MajorSelect']

employee_dataset = pd.DataFrame(dataset[(dataset['EmploymentStatus']=='Employed full-time')|(dataset['EmploymentStatus']=='Employed part-time')],columns=columns_needed)

# Any results you write to the current directory are saved as output.

employee_dataset.head()
# Top 10 countries ranked according to number of respondent

print(employee_dataset['Country'].value_counts()[0:10]);

# age distribution of respondents 

print("Age stats")

print("Mean :",employee_dataset['Age'].mean())

print("Median :",employee_dataset['Age'].median())

print("Mode:",employee_dataset['Age'].value_counts().index[0])

ax=employee_dataset['Age'].hist(bins=100,color='blue',edgecolor='black',figsize=(10,10));

ax.grid(False);

ax.set_xlabel('Age');

ax.set_ylabel('Count');
ax=employee_dataset['FormalEducation'].value_counts().plot.pie(figsize=(10,10),autopct='%.2f%%',shadow=True)

ax.set_ylabel('');
employee_dataset.groupby(['CurrentJobTitleSelect']).agg(lambda x: (x.value_counts().index[0],x.value_counts().iloc[0]*100.0/x.value_counts().sum()))[['FormalEducation']]
employee_dataset.groupby(['CurrentJobTitleSelect']).agg(lambda x: (x.value_counts().index[0],x.value_counts().iloc[0]*100.0/x.value_counts().sum()))[['MajorSelect']]
ax=employee_dataset['LanguageRecommendationSelect'].value_counts(sort=True).plot.pie(figsize=(10,10),autopct='%.2f%%',shadow=True,startangle=45)

ax.set_ylabel('');
employee_dataset.groupby(['CurrentJobTitleSelect']).agg(lambda x: (x.value_counts().index[0],x.value_counts().iloc[0]*100.0/x.value_counts().sum()))[['LanguageRecommendationSelect']]
ax=employee_dataset['MLToolNextYearSelect'].value_counts(sort=True).plot.pie(figsize=(10,10),autopct='%.2f%%',shadow=True,startangle=45)

ax.set_ylabel('');
employee_dataset.groupby(['CurrentJobTitleSelect']).agg(lambda x: (x.value_counts().index[0],x.value_counts().iloc[0]*100.0/x.value_counts().sum()))[['MLToolNextYearSelect']]