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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
file_name = ('coursea_data.csv')

# Reading the dataset using pandas
df = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')

# Renaming the columns to make it more readable
df = df.rename(columns={'course_title': 'Title'
                       ,'Unnamed': 'ID'
                       ,'course_organization': 'Organization'
                       ,'course_Certificate_type': 'Certifiction'
                       , 'course_rating': 'Rating'
                       , 'course_difficulty': 'Difficulty'
                       , 'course_students_enrolled': 'Enrolled'})

# making the Enrolled counts more readable
def func(a):
    if 'k' in a:
        return float(str(a).replace('k', '')) * (10 ** 3)
    if 'm' in a:
        return float(str(a).replace('m', '')) * (10 ** 6)
    else:
        return float(a)

# removing the 'k' from the end of the Enrollmend column and change its type into the float
df['Enrolled'] = df['Enrolled'].apply(func)
# df["Enrolled"] = pd.to_numeric(df.Enrolled, errors='coerce')
# df['Enrolled'] = df['Enrolled'].apply(lambda x: x*1000)

df.head()
# Certifiction destribition over the all courses
df2 =  df.groupby(by=['Certifiction'], as_index=False)['Enrolled'].count().sort_values(by='Enrolled')
plt.figure(figsize=(25, 6))

plt.subplot(131)
plt.bar(df2['Certifiction'], df2['Enrolled'])
plt.subplot(132)
plt.scatter(df2['Certifiction'], df2['Enrolled'])
plt.subplot(133)
plt.plot(df2['Certifiction'], df2['Enrolled'])
plt.suptitle('Categorical Plotting')
plt.show()
df2 =  df.groupby(by=['Certifiction'], as_index=False)['Enrolled'].count().sort_values(by='Enrolled')
df2
# Best Institiuon based on the Enrolled rate
plt.figure(figsize=(50,12))
df3 = df.groupby(df['Organization'], as_index=False)['Enrolled'].count().sort_values(by=['Enrolled'],ascending=False )[:10]

plt.bar(df3['Organization'], df3['Enrolled'], width = 0.5, color='r')

plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=35)

plt.xlabel('List of Organizations', fontsize=30)
plt.ylabel('Enrolled rate', fontsize=30)
plt.title('Top 15 Organization based on the enrolled')

plt.show()
# Organizations with the most courses
plt.figure(figsize=(30, 8))
temp = df.groupby(df['Organization'], as_index=False).agg({'Title': 'count'}).sort_values(by='Title', ascending=False)[:10]
plt.bar(temp.Organization, temp.Title, width=0.5, color='lightblue')

plt.xlabel('Organizations', fontsize=20)
plt.ylabel('Course count', fontsize=20)
plt.title('Top 10 organizations based on the number of the courses', fontsize=20)
plt.xticks(fontsize=15, rotation=90)
plt.show()
# Finding the best Instituions based on the Rating and Enrollment rate
df4 = df.groupby(df['Organization'], as_index=False)['Rating'].agg({'Rating':'mean', 'Enrolled':'sum'}).sort_values(by=['Enrolled'], ascending=False)[0:10]
df4
dfc = df.groupby(df['Title'], as_index=False).agg({'Rating':'mean', 'Enrolled':'sum'}).sort_values(by='Enrolled', ascending=False)[:10]

plt.figure(figsize=(25,6))
plt.bar(dfc['Title'], dfc['Enrolled'], width = 0.5, color='g')

plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)

plt.xlabel('List of Organizations')
plt.ylabel('Enrolled rate')
plt.title('Top 10 Courses on the Coursera platform')

plt.show()
ax = df['Rating'].value_counts().plot(kind='bar', figsize=(25,6), color="dodgerblue", fontsize=12)
ax.set_alpha(0.8)
ax.set_title("Rating Percentage", fontsize=18)
ax.set_ylabel("Count", fontsize=18)
ax.set_xlabel('Rating', fontsize=18)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x(), i.get_height()+.5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,
                color='black')
# Top 10 courses based on the number of enrollments
df5 = df.groupby(df['Title'], as_index=False).agg({'Rating':'mean', 'Enrolled':'sum'}).sort_values(by=['Enrolled'], ascending=False)[0:10]
df5
# Ploting the 10 courses based on the number on enrollments
df5 = df.groupby(df['Title'], as_index=False).agg({'Rating':'mean', 'Enrolled':'sum'}).sort_values(by=['Enrolled'], ascending=False)[0:10]

plt.figure(figsize=(20, 5))
plt.bar(df5['Title'], df5['Enrolled'], width = 0.5, color='dimgrey')

plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)

plt.xlabel('Courses')
plt.ylabel('Enrolled')
plt.title('Top 10 Courses on the Coursera platform')

plt.show()
# Ploting the destribition of course Difficulty
course_Certificate_type_vs_course_difficulty=df.groupby(['Certifiction', 'Difficulty'])['Title'].count()
course_Certificate_type_vs_course_difficulty = course_Certificate_type_vs_course_difficulty.unstack().fillna(0)
ax = (course_Certificate_type_vs_course_difficulty).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
# The chart of the number of enrollemnts based on the difficulty rate
dfd = df.groupby(df['Difficulty'], as_index=False).agg({ 'Title':'count', 'Enrolled':'sum'}).sort_values(by='Enrolled')
# dfd['Enrolled'].round(100)

plt.figure(figsize=(25, 6))

plt.bar(dfd.Difficulty, dfd.Enrolled, width= 0.5, color='lightblue')

plt.xticks(fontsize=20)

plt.xlabel('Difficulty of courses', fontsize=15)
plt.ylabel('The number of enrollments based on the difficulty of courses', fontsize=13)
plt.title('The chart of difficulty based the enrollments', fontsize=30)

plt.show()