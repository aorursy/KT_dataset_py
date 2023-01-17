# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
course=pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
course.head()  #quick look about the information of the csv
course.info()   #about the missing data
# deal with the strange data
course=course.rename(columns={'Unnamed: 0':'Uid'})    #rename the userid
course['course_students_enrolled'] = course['course_students_enrolled'].str.replace('k','')     #student_enrolled units
m = course['course_students_enrolled'].str.endswith('m')
course.loc[m, 'course_students_enrolled'] = pd.to_numeric(course.loc[m, 'course_students_enrolled'].str.replace('m', ''))*1000
course['course_students_enrolled'] = pd.to_numeric(course['course_students_enrolled'])
course.head()
#course_Certificate_type  -- distribution
course['course_Certificate_type']=course['course_Certificate_type'].map({'COURSE':'Course','SPECIALIZATION':'Spacialization','PROFESSIONAL CERTIFICATE':'Professional'}) 
ax1=course['course_Certificate_type'].value_counts().plot(kind='pie',autopct='%.2f%%',fontsize=12)
ax1.set_title('course_Certificate_type',fontsize=16)
ax1.set_ylabel('')
# most courses are course type
#course_difficulty --  distribution
sns.catplot(x='course_Certificate_type',hue='course_difficulty',data=course,kind='count',palette='husl')
plt.ylabel('course count',fontsize=14)
plt.title('Relation between course_type and difficulty')
# pivot_table about course_Certificate_type  and  course_difficulty
course_pt=pd.pivot_table(course,index=['course_difficulty','course_Certificate_type'],values=['Uid','course_rating','course_students_enrolled'],aggfunc={'Uid':'count','course_rating':'mean','course_students_enrolled':'sum'})
course_ct=pd.pivot_table(course,index=['course_difficulty'],values=['Uid'],aggfunc='count')
course_pt['Uid_ratio']=course_pt['Uid']/course_ct['Uid']
course_pt2=pd.pivot_table(course,index=['course_Certificate_type','course_difficulty'],values=['Uid','course_rating','course_students_enrolled'],aggfunc={'Uid':'count','course_rating':'mean','course_students_enrolled':'sum'})
course_pt2['Uid_ratio']=course_pt2['Uid']/course_ct['Uid']
course_pt2
#course_difficuly &course number
fig=plt.figure(figsize=(20,10))
fig.set(alpha=0.2)
ax1=fig.add_subplot(241)
plt.pie(x=course_pt.loc['Beginner']['Uid_ratio'],labels=course_pt.loc['Beginner'].index,autopct='%.2f%%')
ax1.set_title('Beginner')

ax2=fig.add_subplot(242)
plt.pie(x=course_pt.loc['Intermediate']['Uid_ratio'],labels=course_pt.loc['Intermediate'].index,autopct='%.2f%%')
ax2.set_title('Intermediate')

ax3=fig.add_subplot(243)
plt.pie(x=course_pt.loc['Mixed']['Uid_ratio'],labels=course_pt.loc['Mixed'].index,autopct='%.2f%%')
ax3.set_title(' Mixed')

ax4=fig.add_subplot(244)
plt.pie(x=course_pt.loc['Advanced']['Uid_ratio'],labels=course_pt.loc['Advanced'].index,autopct='%.2f%%')
ax4.set_title('Advanced')

plt.show()
#Relation between type, difficulty and rating
sns.catplot(x='course_Certificate_type',y='course_rating',kind='bar',data=course,ci=None,palette='husl')
plt.ylabel('course rating',fontsize=14)
plt.ylim(4.4,4.8)
plt.title('Relation between type, difficulty and rating')

sns.catplot(x='course_Certificate_type',y='course_rating',hue='course_difficulty',data=course,palette='husl')
plt.ylabel('course rating',fontsize=14)
plt.title('Relation between type, difficulty and rating')

sns.catplot(x='course_Certificate_type',y='course_rating',hue='course_difficulty',data=course,kind='point',ci=None,linestyles=':',join=True,palette='husl',fontsize=16)
plt.ylabel('course rating',fontsize=14)
plt.title('Relation between type, difficulty and rating')
#Relation between type and enrolled
fig=plt.figure(figsize=(15,4))
fig.set(alpha=0.2)
fig.add_subplot(121)
plt.bar(x='course_Certificate_type',height='course_students_enrolled',data=course)
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between type and enrolled')

#Relation between difficulty and enrolled
fig.add_subplot(122)
plt.bar(x='course_difficulty',height='course_students_enrolled',data=course)
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between difficulty and enrolled')

#Relation between type, difficulty and enrolled
sns.catplot(x='course_Certificate_type',y='course_students_enrolled',hue='course_difficulty',data=course,palette='husl')
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between type, difficulty and enrolled')
#Relation between type, difficulty and enrolled
sns.barplot(x='course_rating',y='course_students_enrolled',data=course,palette='husl',estimator=sum,ci=None)
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between rating and enrolled')

#Relation between rating and enrolled
sns.catplot(x='course_rating',y='course_students_enrolled',data=course,palette='husl')
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between rating and enrolled')

#Relation between rating, difficulty and enrolled
sns.catplot(x='course_rating',y='course_students_enrolled',hue='course_difficulty',data=course,palette='husl')
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between rating, difficulty and enrolled')

#Relation between rating,type and enrolled
sns.catplot(x='course_rating',y='course_students_enrolled',hue='course_Certificate_type',data=course,palette='husl')
plt.ylabel('course enrolledg',fontsize=14)
plt.title('Relation between rating,type and enrolled')
#pivot table about course_organization, rating, student_enrolled  
course_pt3=pd.pivot_table(course,index=['course_organization'],values=['Uid','course_rating','course_students_enrolled'],aggfunc={'Uid':'count','course_rating':'mean','course_students_enrolled':'sum'}).sort_values(by='Uid',ascending=False)
course_ct3=pd.pivot_table(course,index=['course_organization','course_difficulty'],values=['Uid'],aggfunc='count').sort_values(by='Uid',ascending=False)
pt3=course_pt3[:10]
print(course_ct3[:10])
print(pt3)
#organizations 
fig=plt.figure(figsize=(20,6))
fig.set(alpha=0.2)
ax1=fig.add_subplot(131)
plt.barh(pt3.index.tolist(),pt3['Uid'])
ax1.set_title('course_count')

ax2=fig.add_subplot(1,3,2,sharey=ax1)
plt.barh(pt3.index.tolist(),pt3['course_students_enrolled'])
ax2.set_title('students_enrolled_count')

ax3=fig.add_subplot(1,3,3,sharey=ax1)
plt.plot(pt3['course_rating'],pt3.index.tolist(),marker='o')
ax3.set_title('course_rating')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
# key_words & orgnazations
key_words=['Python','Data','Machine Learning','AI','Statistics','Management','Business','Strategic','Introduction','Analytics','English']
fig=plt.figure(figsize=(20,10))
fig.set(alpha=0.2)
def pplot(j,i):
    plt.subplot(3,4,j+1)
    course['course_organization'][course['course_title'].str.contains(i)].value_counts()[:3].plot(kind='bar')
    plt.xticks(rotation=30)
    plt.title(i) 
    plt.subplots_adjust(hspace=1, wspace=0.3)
    
for j,i in enumerate(key_words):
    pplot(j,i)
#key_words & course
key_words=['Python','Data','Machine Learning','AI','Statistics','Management','Business','Strategic','Introduction','Analytics','English']
d={}
for i in key_words:
    key_count=course[course['course_title'].str.contains(i)]['course_title'].count()
    enrolled_count=course['course_students_enrolled'][course['course_title'].str.contains(i)].sum()
    d[i]=[key_count,enrolled_count]
    a = sorted(d.items(), key=lambda x: x[1][0], reverse=False)
    b = sorted(d.items(), key=lambda x: x[1][1], reverse=False)
fig=plt.figure(figsize=(20,6))
fig.set(alpha=0.2)
ax1=fig.add_subplot(121)
for j in a:
    #print(list(j))
    ax1=plt.barh(list(j)[0],list(j)[1][0])
    plt.xlabel('course_number')
    plt.title('key_word & course_number')
ax2=fig.add_subplot(122)
for k in b:
    #print(list(k))
    ax2=plt.barh(list(k)[0],list(k)[1][1])
    plt.xlabel('student_enrolled')
    plt.title('key_word & student_enrolled')
#most student_enrolled courses top 10
course_pt4=pd.pivot_table(course,index=['course_title'],values=['course_students_enrolled','course_rating'],aggfunc={'course_students_enrolled':'sum','course_rating':'mean'}).sort_values(by='course_students_enrolled',ascending=False)
course_most=course_pt4[:10]
course_most
plt.barh(course_most.index,course_most['course_students_enrolled'],color='tan')
plt.axvline(x=1000, color='red', linestyle='--')
