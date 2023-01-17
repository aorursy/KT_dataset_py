#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# let (to be used later)
passmark=40
## Reading the dataset
df=pd.read_csv('../input/StudentsPerformance.csv')
## getting the top of dataset
df.head()
## getting the features of dataset
df.columns
## finding the mean,variance,count etc of data
df.describe()
#DISRTRIBUTION OF MARKS

sns.distplot(df['math score'])


sns.distplot(df['reading score'])
sns.distplot(df['writing score'])
## Plot the bar plot to visualize the effect of gender on the scores of students
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
sns.barplot(x = 'gender', y = 'reading score', data = df)

plt.subplot(1,3,2)
sns.barplot(x = 'gender', y = 'writing score', data = df)

plt.subplot(1,3,3)
sns.barplot(x = 'gender', y = 'math score', data = df)

plt.tight_layout()

## Plotting bar plot to visualize the effect of race on marks of students

plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
sns.barplot(x='race/ethnicity',y='math score',data=df)

plt.subplot(1,3,2)
sns.barplot(x='race/ethnicity',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot (x='race/ethnicity',y='writing score',data=df)

plt.show()

## plotting bar plot to visualize the effect of parents educational qualification on the marks of students
plt.figure(figsize=(13,5))
plt.subplot(1,3,1)
sns.barplot(x='parental level of education', y= 'math score', data = df )
plt.xticks(rotation = 90)

plt.subplot(1,3,2)
sns.barplot(x='parental level of education', y= 'writing score', data=df)
plt.xticks(rotation = 90)

plt.subplot(1,3,3)
sns.barplot(x='parental level of education',y='reading score', data=df)
plt.xticks(rotation = 90)


## plotting bar plot to visualize the effect of lunch on marks of students
plt.figure(figsize=(13,8))
plt.subplot(1,3,1)
sns.barplot(x='lunch', y= 'math score', data=df)

plt.subplot(1,3,2)
sns.barplot(x='lunch',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot(x='lunch',y='writing score', data=df)


### Plotting bar plot to visualize the effect of test prepration course
plt.figure(figsize=(13,8))
plt.subplot(1,3,1)
sns.barplot(x='test preparation course',y='math score' ,data=df)

plt.subplot(1,3,2)
sns.barplot(x='test preparation course',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot(x='test preparation course',y='writing score', data=df)
## How many students have passed in Maths?
df['Math_PassStatus'] = np.where(df['math score']<passmark, 'F', 'P')
df.Math_PassStatus.value_counts()


## How many students have passed in Reading?
df['Reading_PassStatus'] = np.where(df['reading score']<passmark, 'F', 'P')
df.Reading_PassStatus.value_counts()

## How many Students have passed in Writing?
df['Writing_PassStatus'] = np.where(df['writing score']<passmark,'F','P')
df.Writing_PassStatus.value_counts()
## Students who have passed in all subjects.
df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()
## Finding the percentage of marks
df['Total_Marks'] = df['math score']+df['writing score']+df['reading score']
df['Percentage']= df['Total_Marks']/3
### Effect of gender on marks
gender_effect = df.groupby('gender')['Percentage'] \
                        .aggregate(['mean', 'var', 'count']) \
                        .replace(np.NaN, 0) \
                        .sort_values(['mean', 'var'], ascending=[False, False])
gender_effect.head()
### Effect of race of students
mean_marks_race = df.groupby('race/ethnicity')['Percentage'] \
                    .aggregate(['mean','var', 'count']) \
                    .replace(np.NaN, 0) \
                    .sort_values(['mean','var'],ascending=[False,False])
mean_marks_race.head()
## Effect of parents qualification on marks
effect_parental_degree = df.groupby('parental level of education')['Percentage'] \
                               .aggregate(['mean','var','count']) \
                               .replace(np.NaN,0) \
                               .sort_values(['mean','var'],ascending=[False,False]) 

effect_parental_degree.head()

### Effect of type of lunch on marks 
effect_lunch = df.groupby('lunch')['Percentage'] \
                     .aggregate(['mean','var','count']) \
                     .replace(np.NaN,0) \
                     .sort_values(['mean','var'],ascending=[False,False])
effect_lunch.head()
### Effect of test preparation on marks
effect_testprep= df.groupby('test preparation course')['Percentage'] \
                       .aggregate(['mean','var','count']) \
                       .replace(np.NaN,0) \
                       .sort_values(['mean','var'],ascending=[False,False])
effect_testprep.head()
