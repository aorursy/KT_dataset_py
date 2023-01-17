# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_public.csv")

df_schema=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_schema.csv")
print(df.shape[0])

print(df.shape[1])
df.head()
df_schema[df_schema['Column'].str.contains('JobEmailPriorities')]
df.head(20)
df.shape
df['Age']
df.describe(percentiles = None)
df["Gender"].describe()
df["Gender"].astype('category').cat.codes
c = df.Gender.astype('category')



d = dict(enumerate(c.cat.categories))

d
pd.get_dummies(df['Country'])
df['OpenSource']
df["FormalEducation"].values
df[df["Hobby"]==True]["Hobby"].sum()
df['Hobby'].values.sum()
df['Count'] = (df[['Hobby']] == 'No').sum(axis=1)

print (df)
df[df["Hobby"] == "Yes"][["Hobby"]]
df["Country"].unique()
len(df["Country"].unique())
df[df["Student"] == 'No']
df[df["Student"] == 'No'].count()
df["Respondent"].max()
df["Respondent"].min()
df["Respondent"].std()
df[df["Country"] == "India"][["Gender", "Employment"]]
(df["Respondent"] > 0).any()
df[['Country','CompanySize']]
sns.countplot(df['Hobby'])
sns.countplot(df['Gender'])
sns.countplot(df['SurveyTooLong'])
sns.countplot(df['Student'])
df.CompanySize.value_counts(sort=False).plot.pie()
df.EducationParents.value_counts(sort=False).plot.pie()
plt.figure(figsize=(8,8))

g=sns.countplot(df['CompanySize'])

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_xlabel("Response")

g.set_ylabel("Count")

g.set_title("How do the respondents feel about the survey ?")# Not a right way to compare
plt.figure(figsize=(8,8))

g=sns.countplot(df['Gender'])

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_xlabel("Response")

g.set_ylabel("Count")

g.set_title("How do the respondents feel about the survey ?")# Not a right way to compare
plt.figure(figsize=(8,8))

g=sns.countplot(df['Country'])

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_xlabel("Response")

g.set_ylabel("Count")

g.set_title("How do the respondents feel about the survey ?")# Not a right way to compare
plt.figure(figsize=(8,8))

g=sns.countplot(df['UndergradMajor'])

g.set_xticklabels(g.get_xticklabels(),rotation=90)

g.set_xlabel("Response")

g.set_ylabel("Count")

g.set_title("How do the respondents feel about the survey ?")# Not a right way to compare
df[['Respondent','Gender']]
df['Age'].hist(bins=50)
plt.figure(figsize = (16, 8))



sns.distplot(df["Respondent"])

plt.title("Respondent Histogram")

plt.xlabel("Respondent")

plt.show()
e=df["Age"].astype('category').cat.codes

plt.figure(figsize = (16, 8))



sns.distplot(e)

plt.title("Age Histogram")

plt.xlabel("Age")

plt.show()
e.max()
e.min()
e.std()
e>1
age = np.array(e)

type(age)
age[2]
age[0:100]
np.random.random((4,5))
np.random.random((4, 4)) > 0.25
np.random.random((4,1)).shape
np.arange(10)
np.eye(5)
np.ones((4,3))