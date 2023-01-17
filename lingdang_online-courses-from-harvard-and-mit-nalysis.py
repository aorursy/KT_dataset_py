import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/course-study/appendix.csv')

df.head()
df.shape
df.duplicated().value_counts()
df.isnull().sum()
df[df['Instructors'].isnull()]
df[df['Course Number']=='CS50x']['Instructors']
df['Instructors'].fillna('David Malan',inplace=True)

df['Instructors'].isnull().sum()
#主题

Subject=df.groupby('Course Subject')['Participants (Course Content Accessed)'].sum().sort_values(ascending=False)

Subject=Subject.to_frame()

Subject.head()
#课程

course=df.groupby('Course Title')['Participants (Course Content Accessed)'].sum().sort_values(ascending=False)

course=course.to_frame()

course.head()
plt.figure(figsize=(10,15))

# 可视化主题-子图1

plt.subplot(211)

label=Subject.index

value=Subject['Participants (Course Content Accessed)']

plt.pie(value,labels=label,autopct='%.1f%%',)

plt.title('Course Subject')



# 可视化前10的课程-子图2

plt.subplot(212)

course_top10=course[0:10]

plt.plot(course_top10)

plt.title('top10--Course Title')

plt.xticks(rotation=80)

plt.xlabel('Course Title')

plt.ylabel('学生数')

#学校课程欢迎度

sc=pd.pivot_table(df,index='Institution',columns='Course Subject',values='Participants (Course Content Accessed)',aggfunc='sum')

sc
#学校开设课程分布

sc_sub=pd.pivot_table(df,index='Institution',columns='Course Subject',values='Participants (Course Content Accessed)',aggfunc='count')

sc_sub
plt.figure(figsize=(15,8))

#学校课程学生数-子图1

plt.subplot(121)

index=sc.index

y1=sc['Computer Science']

y2=sc['Government, Health, and Social Science']

y3=sc['Humanities, History, Design, Religion, and Education']

y4=sc['Science, Technology, Engineering, and Mathematics']

plt.bar(index,y1,label='workday')

plt.bar(index,y2,bottom=y1,label='Government, Health, and Social Science')

plt.bar(index,y3,bottom=y1+y2,label='Humanities, History, Design, Religion, and Education')

plt.bar(index,y4,bottom=y1+y2+y3,label='Science, Technology, Engineering, and Mathematics')

plt.title('HarvardX&MITx---Participants')

plt.legend()

#学校开设课程-子图2

x=sc_sub.index

yy1=sc_sub['Computer Science']

yy2=sc_sub['Government, Health, and Social Science']

yy3=sc_sub['Humanities, History, Design, Religion, and Education']

yy4=sc_sub['Science, Technology, Engineering, and Mathematics']

plt.subplot(122)

plt.bar(x,yy1)

plt.bar(x,yy2,bottom=yy1)

plt.bar(x,yy3,bottom=yy1+yy2)

plt.bar(x,yy4,bottom=yy1+yy2+yy3)

plt.title('HarvardX&MITx---Course Subject  ')

plt.legend()



stu=df[['% Audited','% Certified']]

stu.describe()
plt.figure(figsize=(10,5))

plt.boxplot([stu['% Audited'],stu['% Certified']])   

plt.xticks([1,2],['% Audited','% Certified']) 

plt.title('课程完成度')

df.columns
play=df[['Course Title','Total Course Hours (Thousands)','Participants (Course Content Accessed)','% Played Video','% Certified']]

play.describe()
plt.figure(figsize=(10,5))

plt.hist(df['Total Course Hours (Thousands)'],bins=15)  

plt.xlabel('Total Course Hours (Thousands)')
cour_play=play.groupby(by='Course Title')['Total Course Hours (Thousands)','% Certified'].mean()

cour_play.head()
corr=play.corr()

print(corr['% Certified'])

plt.figure(figsize=(10,6))

plt.scatter(cour_play['Total Course Hours (Thousands)'],cour_play['% Certified'])
corr=df.corr()

corr

corr['% Certified'].sort_values()

# import seaborn as sns

# sns.pairplot(df)
1# 年龄特征

df['Median Age'].describe()
plt.boxplot(df['Median Age'])
#2、性别特征

sex=df[['% Male','% Female']]

sex.describe()
label=sex.columns  

plt.pie(sex.mean(),labels=label,autopct='%.1f%%')

plt.title('Course Subject')
df.groupby(by='Course Subject')['% Male','% Female'].mean()
#3、学历特征

educational=df["% Bachelor's Degree or Higher"]

educational.describe()