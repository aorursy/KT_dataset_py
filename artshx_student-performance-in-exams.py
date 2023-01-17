# Kaggle repo
# https://www.kaggle.com/spscientist/student-performance-in-exams 
# by Arthur H. M. da Silva - artshx
# 2018/december/19

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set(style="darkgrid")
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.isna().any()
sns.countplot(df['gender'], order=['male','female'])
sns.countplot(df['gender'], order=['male', 'female'], hue=df['race/ethnicity'])
labels = ['group A', 'group B','group C','group D','group E']
num_A = (df['race/ethnicity'] == 'group A').sum()
num_B = (df['race/ethnicity'] == 'group B').sum()
num_C = (df['race/ethnicity'] == 'group C').sum()
num_D = (df['race/ethnicity'] == 'group D').sum()
num_E = (df['race/ethnicity'] == 'group E').sum()
num_occorrences = [num_A, num_B, num_C, num_D, num_E]

#plt.title('% of each Race/Ethnicity')
plt.figure(figsize=(15,8))
#explode = (0, 0, 0, 0, 0)

plt.pie(num_occorrences, labels=labels, autopct='%1.1f%%')
#df['race/ethnicity'].values
num_std = (df['lunch'] == 'standard').sum()
num_red = (df['lunch'] == 'free/reduced').sum()
arr_lunch = [num_std, num_red]
plt.title("Lunch - Standard VS Free/Reduced")
plt.pie(arr_lunch, labels=['standard','free/reduced'], autopct='%1.1f%%')
df['overall score'] = (df['math score'] + df['reading score'] + df['writing score'])
df.head()
males = (df['gender'] == 'male')
male_avg = df[males][['math score', 'reading score', 'writing score']].mean()
male_avg

females = (df['gender'] == 'female')
female_avg = df[females][['math score', 'reading score', 'writing score']].mean()
female_avg
#fig = plt.figure()
#ax = fig.add_subplot(1, 3, 1)
#ax.hist(df[males]['math score'], clip_on=False)
#ax.set_title('Males X Math Score')

#ax = fig.add_subplot(1, 3, 2)
#ax.hist(df[males]['reading score'], clip_on=False)
#ax.set_title('Males X Reading Score')

#ax = fig.add_subplot(1, 3, 3)
#ax.hist(df[males]['writing score'], clip_on=False)
#ax.set_title("Malex X Writing Score")

#plt.tight_layout()
#plt.show()
df_table = pd.DataFrame({'math score avg': [male_avg[0], female_avg[0]], 'reading score avg': [male_avg[1], female_avg[1]], 
                         'writing score avg':[male_avg[2], female_avg[2]], "gender":['male', 'female']})
df_table.set_index("gender", inplace=True)
#plt.table(cellText=df_table, rowLabels=['male','female'], colLabels=('math score avg','reading score avg','writing score avg'))

df_table
df['overall score'].hist(bins=30)
test_completed = (df['test preparation course'] == 'completed')
test_completed_avg = df[test_completed]['overall score'].mean()
test_n_completed_avg = df[~test_completed]['overall score'].mean()

#test_completed_avg = df[test_completed]['math score'].mean()
#test_n_completed_avg = df[~test_completed]['math score'].mean()
test_completed_avg, test_n_completed_avg
increased_grade = (test_completed_avg * 100) / test_n_completed_avg - 100
increased_grade
df['parental level of education'].unique()
plt.tight_layout()
plt.figure(figsize=(15,8))
parent_education = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
sns.countplot(df['parental level of education'], order=parent_education)
education = list()
ed1 = (df['parental level of education'] == parent_education[0])
ed2 = (df['parental level of education'] == parent_education[1])
ed3 = (df['parental level of education'] == parent_education[2])
ed4 = (df['parental level of education'] == parent_education[3])
ed5 = (df['parental level of education'] == parent_education[4])
ed6 = (df['parental level of education'] == parent_education[5])
education.append(df[ed1]['overall score'].mean())
education.append(df[ed2]['overall score'].mean())
education.append(df[ed3]['overall score'].mean())
education.append(df[ed4]['overall score'].mean())
education.append(df[ed5]['overall score'].mean())
education.append(df[ed6]['overall score'].mean())
plt.figure(figsize=(15,8))
plt.title('parental level of education vs student overall score')
plt.plot(parent_education, education)
df_grades = df[['math score', 'reading score', 'writing score']]
df_grades.corr('spearman')
groups = ['group A','group B','group C','group D','group E']
group_A = (df['race/ethnicity'] ==  groups[0])
group_B = (df['race/ethnicity'] ==  groups[1])
group_C = (df['race/ethnicity'] ==  groups[2])
group_D = (df['race/ethnicity'] ==  groups[3])
group_E = (df['race/ethnicity'] ==  groups[4])
group_mean = list()
group_mean.append(df[group_A]['overall score'].mean())
group_mean.append(df[group_B]['overall score'].mean())
group_mean.append(df[group_C]['overall score'].mean())
group_mean.append(df[group_D]['overall score'].mean())
group_mean.append(df[group_E]['overall score'].mean())
fig, ax = plt.subplots()
plt.title('Overall score x Race/Ethnicity')
ax.barh(groups, group_mean, color="blue")
for i, v in enumerate(group_mean):
    ax.text(10, i , str("{:.2F}".format(v)), color='white', fontweight='bold')
#plt.bar(groups, group_mean)
