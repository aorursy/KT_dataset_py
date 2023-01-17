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
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset
df = pd.read_csv("../input/coursera-course-dataset/coursea_data.csv",index_col=0)

df.head()
df.info()
df.columns
df.isnull().sum()
labels=['COURSE','SPECIALIZATION','PROFESSIONAL CERTIFICATE']
data=df.course_Certificate_type.value_counts()
print(data)

plt.figure(figsize=(20,6))
plt.pie(data,labels=labels,explode=[0.01,0.01,0.01],autopct='%0.02f%%')
plt.title("Course certification Analysis",fontsize=16)

plt.legend()
plt.show()
data=df.course_difficulty.value_counts()
print(data)
labels=['Beginner','Intermediate','Mixed','Advanced']
plt.figure(figsize=(20,6))
plt.pie(data,labels=labels,autopct='%0.02f%%',explode=[0.01,0.01,0.01,0.01])
plt.title("Course difficulty Analysis",fontsize=16)

plt.legend()
plt.show()
plt.figure(figsize=(10,7))
plt.title("Course Difficulty analysis",fontsize=16)
ax=sns.countplot(x="course_difficulty",data=df)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
plt.title("course Certificate type Analysis")

ax=sns.countplot(x="course_Certificate_type",data=df)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
# convert enrolled to no.
df['course_students_enrolled']=df['course_students_enrolled'].str.replace('k','*1000')
df['course_students_enrolled']=df['course_students_enrolled'].str.replace('m','*10000')
df['course_students_enrolled']=df['course_students_enrolled'].map(lambda x:eval(x))
df['course_students_enrolled']
plt.figure(figsize=(10,7))
plt.title("course Difficulty and Course Student Enrolled",fontsize=25)
plt.xlabel("course Difficulty",fontsize=20)
plt.ylabel("Student Enrolled",fontsize=20)
ax=sns.barplot(df.course_difficulty,df.course_students_enrolled)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
plt.title("course Difficulty and Course Student Enrolled",fontsize=16)
ax=sns.countplot(x='course_Certificate_type',data=df,hue='course_difficulty')
data=df.nlargest(10,'course_students_enrolled')
sns.set_style("darkgrid")
plt.figure(figsize=(10,7))
plt.title("Top 10 Course Using Enrolled",fontsize=25)
plt.xlabel("Top 10 Course",fontsize=20)
plt.ylabel("Student Enrolled",fontsize=20)

ax=sns.barplot(data.course_title,data.course_students_enrolled)
ax.set_xticklabels(rotation=75,labels=data.course_title)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(20,6))
sns.set_style("darkgrid")
Top10=df.course_organization.value_counts().sort_values(ascending=False).head(10)
Top10
ax=sns.barplot(Top10.index,Top10.values)
ax.set_xticklabels(rotation=30,labels=Top10.index)
plt.title("Top 10 Course Organization",fontsize=25)
plt.xlabel("Organizations",fontsize=20)
plt.ylabel("No Of",fontsize=20)


for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
    
plt.show()