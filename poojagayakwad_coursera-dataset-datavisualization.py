# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
coursera = pd.read_csv("/kaggle/input/coursera-course-dataset/coursea_data.csv")

coursera.columns
coursera.head(3)
# dropping course_id column
courses = coursera.drop(['Unnamed: 0'], axis=1)
courses['count']=1
# fonts for labels and titles of various plots
font_title={'fontfamily':'monospace','fontweight':'bold','fontsize':20}
font_label={'fontfamily':'monospace', 'fontsize':12}
types = courses.groupby(['course_Certificate_type']).sum().sort_values(by='course_Certificate_type')['count']

type_list =list(courses['course_Certificate_type'].unique())

type_list.sort()
# plotting a pie chart
plt.pie(types, labels = type_list, autopct = '%.2f%%',radius=2, explode = [0.1]*3, startangle =180)
plt.show()
# creating a new column 'enrolled' which indicates number of students enrolled in that course (in float)
courses['enrolled'] = courses['course_students_enrolled'].str[:-1]
courses['enrolled'] = courses['enrolled'].astype('float')*1000


# applying groupby to calculate students enrolled in different course type

num_students = courses.groupby('course_Certificate_type').sum()['enrolled']

labels=[]
for i in range(3):
    labels.append(str(type_list[i])+'  ('+   str( list(num_students)[i]/1000000 )+"  million )")
 

# plotting a pie chart
plt.pie(num_students, labels = labels, autopct ='%.2f%%', startangle = 150, radius= 2, explode = [0.1]*3)
plt.show()

best_courses =pd.DataFrame(columns=['course_title','course_organization','course_Certificate_type','course_students_enrolled','enrolled'] )

for ctype in type_list:
    data = courses[['course_title','course_organization','course_Certificate_type','course_students_enrolled','enrolled']]
    data = data.loc[courses['course_Certificate_type'] == ctype].sort_values(by='enrolled', ascending =False).head(5)
    best_courses = pd.concat([best_courses, data])

    
fig= plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
sns.barplot(data= best_courses, y='course_title', x='enrolled',ax=ax ,hue='course_Certificate_type', orient='h',dodge=False)
sns.set(style="whitegrid")
universities = ['Stanford University', 'University of London', 'University of Pennsylvania',  'Columbia University', 'Yale University']


univ_wise = pd.DataFrame(columns = ['course_title', 'course_organization','course_rating','course_students_enrolled','enrolled'])

for univ in universities:
    data=courses[['course_title', 'course_organization','course_rating','course_students_enrolled','enrolled']]
    data =data.loc[(courses.course_organization==univ) & (courses.course_rating >=4.5)].sort_values(by='enrolled', ascending=False).head(3)
    univ_wise = pd.concat([univ_wise, data])

univ_wise
companies =['IBM', 'Amazon Web Services', 'Cisco',  'Google', 'Google Cloud', 'Palo Alto Networks', 'VMware']

comp_wise = pd.DataFrame(columns = ['course_title', 'course_organization','course_rating','course_students_enrolled','enrolled'])

for comp in companies:
    data=courses[['course_title', 'course_organization','course_rating','course_students_enrolled','enrolled']]
    data =data.loc[(courses.course_organization==comp) & (courses.course_rating >=4.5)].sort_values(by='enrolled', ascending=False).head(3)
    comp_wise = pd.concat([comp_wise, data])
    
comp_wise
bins = [3.0, 3.5, 4.0, 4.5, 5.0]
plt.hist(courses['course_rating'], bins)
plt.xticks(bins)

plt.xlabel("Course ratings", fontdict=font_label)
plt.ylabel("no. of courses", fontdict =font_label)

plt.show()

difficulty = courses['course_difficulty'].sort_values().unique()
course= courses.loc[courses['course_Certificate_type']=='COURSE'].groupby('course_difficulty').count()['count']

plt.pie(course, labels=difficulty, autopct='%.2f%%', radius=2, explode=[0.1]*4)
plt.show()

course
labels= ['Beginner', 'Intermediate']
course= courses.loc[courses['course_Certificate_type']=='PROFESSIONAL CERTIFICATE'].groupby('course_difficulty').count()['count']

plt.pie(course, labels=labels, autopct='%.2f%%', radius=2, explode=[0.1]*2)
plt.show()
course
labels = ['Advanced','Beginner','Intermediate']
course= courses.loc[courses['course_Certificate_type']=='SPECIALIZATION'].groupby('course_difficulty').count()['count']

plt.pie(course, labels=labels, autopct='%.2f%%', radius=2, explode=[0.1]*3)
plt.show()
course