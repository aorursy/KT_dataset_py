import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style("dark")

plt.rcParams['figure.figsize'] = (12, 9)
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

df.head()
df.shape
df.describe()
df.isna().any()
sns.heatmap(df.corr(), annot=True)
df.columns
df_gender = df.loc[:, ['gender', 'math score', 'reading score','writing score']]

df_gender_per = df_gender.groupby(["gender"]).mean()

df_gender_per.plot(kind='barh')

plt.title("Avg. score with respect to gender")

plt.xlim(0, 100)
df['avg_score'] = round((df['math score']+df['reading score']+df['writing score'])/3,0)

df.head()



df_test_course = df.loc[:, ['gender', 'avg_score', 'test preparation course']]

df_test_course = df_test_course.groupby(["test preparation course", "gender"], as_index=False).mean()



sns.barplot(x='test preparation course',y='avg_score',data=df_test_course,palette='viridis', hue='gender')

plt.xticks(rotation=45,ha='right')

plt.ylim(0, 100)

plt.title("test preparation course vs Avg. score")

plt.tight_layout()
df_test_prep_course = df.loc[:, ['test preparation course', 'math score', 'reading score','writing score']]

df_test_prep_course = df_test_prep_course.groupby(["test preparation course"]).mean()

df_test_prep_course.plot(kind='barh')

plt.title("Avg. score with respect to test preparation course")

plt.xlim(0, 100)
df_lunch = df.loc[:, ['lunch', 'math score', 'reading score','writing score']]

df_lunch = df_lunch.groupby(["lunch"]).mean()

df_lunch.plot(kind='barh')

plt.title("Avg. score with respect to lunch type")

plt.xlim(0, 100)
df.columns
df_race = df.loc[:, ['race/ethnicity', 'math score', 'reading score','writing score']]

df_race = df_race.groupby(["race/ethnicity"]).mean()

df_race.plot(kind='barh')

plt.title("Avg. score with respect to race/ethnicity")

plt.xlim(0, 100)
df_race_gender = df.loc[:, ['race/ethnicity', 'gender', 'avg_score']]

df_race_gender = pd.pivot_table(df_race_gender, index=["race/ethnicity"], values=["avg_score"], columns=["gender"])

# df_race_gender.head()

df_race_gender.plot(kind='barh')

plt.title("Avg. score with respect to race vs gender")

plt.xlim(0, 100)
df_parenat_edu = df.loc[:, ['parental level of education', 'avg_score', 'gender']]

df_parenat_edu = pd.pivot_table(df_parenat_edu, index=["parental level of education"], values=["avg_score"], columns=["gender"])

df_parenat_edu.plot(kind='barh')

plt.title("Avg. score with respect to parental level of education")

plt.xlim(0, 100)
df_race_parent_edu = df.loc[:, ['race/ethnicity', 'parental level of education', 'avg_score']]

df_race_parent_edu = pd.pivot_table(df_race_parent_edu, index=["parental level of education"], values=["avg_score"], columns=["race/ethnicity"])



df_race_parent_edu.plot(kind='barh')

plt.title("Avg. score with respect to race vs parental level of education")

plt.xlim(0, 100)