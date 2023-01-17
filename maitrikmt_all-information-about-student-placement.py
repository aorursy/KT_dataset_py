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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
# information of the Dataset
df.info()
# check null values in dataset
count_missing=df.isnull().sum()
percent_missing=count_missing*100/df.shape[0]
missing_value=pd.DataFrame({'Count_Missing':count_missing,
                            'percent_missing':percent_missing})
missing_value
#Here we can see 31% missing value in salary column
plt.figure(figsize=(12,7))
data=df.gender.value_counts()
labels=['Male','Female']
plt.title("percentage of Male or Female",fontsize=18)
plt.pie(data=data,x=data.values,autopct="%.2f%%",labels=labels)
plt.show()
plt.figure(figsize=(12,7))
ax=sns.countplot(x="gender",data=df)
plt.title("Gender wise students",fontsize=20)

plt.xlabel("Gender",fontsize=18)
plt.ylabel("Count",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
ax=sns.countplot(x="hsc_s",data=df)
plt.title("student Belong which fields in 12th",fontsize=20)

plt.xlabel("Fields",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
ax=sns.countplot(x="degree_t",data=df)
plt.title("complete Degree in field",fontsize=20)

plt.xlabel("Degree",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
ax=sns.countplot(x="specialisation",data=df)
plt.title("student Belong which specialisation ",fontsize=20)

plt.xlabel("specialisation ",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.figure(figsize=(10,7))
ax=sns.countplot(x="specialisation",data=df,hue=df.gender)
plt.title("student Belong which specialisation Gender wise",fontsize=20)

plt.xlabel("specialisation ",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.legend(fontsize=22)
plt.show()
plt.figure(figsize=(10,7))
ax=sns.countplot(x="workex",data=df)
plt.title("Student work experience or not",fontsize=20)

plt.xlabel("Experience",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
    
plt.figure(figsize=(10,7))
ax=sns.countplot(x="workex",data=df,hue=df.gender)
plt.title("Student work experience or not gender wise",fontsize=20)

plt.xlabel("Experience",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.legend(fontsize=22)
plt.show()
plt.figure(figsize=(10,7))
ax=sns.countplot(x="status",data=df)
plt.title("NO. of student placed or not",fontsize=20)

plt.xlabel("Placed or not",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')

plt.show()
plt.figure(figsize=(10,7))
ax=sns.countplot(x="status",data=df,hue=df.gender)
plt.title("NO. of Male or Female student placed or not",fontsize=20)

plt.xlabel("Placed or not",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
    
plt.legend(fontsize=22)
plt.show()
plt.figure(figsize=(10,7))
ax=sns.countplot(x="status",data=df,hue=df.workex)
plt.title("NO. of student placed or not",fontsize=20)

plt.xlabel("Placed or not",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
    
plt.legend(fontsize=22)
plt.show()
plt.figure(figsize=(10,7))
ax=sns.countplot(x="degree_t",data=df,hue=df.status)
plt.title("Category wise no. of student placed or not",fontsize=20)

plt.xlabel("Placed or not",fontsize=18)
plt.ylabel("No of student",fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d'% int(p.get_height()),
           fontsize=12,color='blue',ha='center',va='bottom')
plt.legend(fontsize=22)
plt.show()