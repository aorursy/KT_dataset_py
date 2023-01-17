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
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
#This librarys is to work with matrices

import pandas as pd 

# This librarys is to work with vectors

import numpy as np

# This library is to create some graphics algorithmn

import seaborn as sns

# to render the graphs

import matplotlib.pyplot as plt

# import module to set some ploting parameters

from matplotlib import rcParams

# Library to work with Regular Expressions

import re



# This function makes the plot directly on browser

%matplotlib inline



# Seting a universal figure size 

rcParams['figure.figsize'] = 10,8
df.describe()
df.shape
len(df.Location.value_counts())
df.Location.value_counts()
df['PLocation'] = df['Location'].str.split(' |,').str[0]

df['PLocation']
fig,ax = plt.subplots(figsize = (15,120))

sns.countplot(y='PLocation',data = df )

plt.show()
df['Job Experience Required'].value_counts()


df['Job Salary'].value_counts()
len(df['Job Salary'].value_counts())
df['Date crawl']=df['Crawl Timestamp'].str.split(' ').str[0]
fig,ax = plt.subplots(figsize = (15,120))

sns.countplot(x='Date crawl',data = df )

plt.show()
skills=df['Key Skills'].str.split('|')
skills=skills.to_frame()
skills=skills.dropna()
skill={}

for index,row in skills.iterrows():

        for i in row['Key Skills']:

            if i.strip() in skill:

                skill[i.strip()]+=1

            else:

                skill[i.strip()]=1

            print(i.strip())

   

    
skills=pd.DataFrame([skill])
#skills.shape

skills
temp = sorted ( skill)
for i in sorted (skill) : 

    print ((i, temp[i]), end =" ") 
temp
skills=skills.T
new_skills=skills.rename(columns={0:'count'})

new_skills=new_skills.sort_values('count', ascending=False)
top_skill=new_skills[:][:10]
ax=top_skill.plot.bar(y='count',rot=0)
titles=df['Job Title'].str.lstrip().value_counts().to_frame()
len(df['Job Title'].value_counts())
top_title=titles.head(10)
titles.max(axis=1)


ax=top_title.plot.bar(y='Job Title',rot=0,figsize=(25, 10))
df['Job Title']=df['Job Title'].str.lstrip()
skills_req={}



for index,row in top_title.iterrows():

    skills_req[index]=df[df['Job Title']==index]["Key Skills"]

#     print(index,df.loc[row['Job Title'],['Key Skills']],sep=' : ')

#     print('=====================================')


letc=[]

for key,value in skills_req.items():

    for i in value:

        try:

            #print(i.split('|'))

            skills_req[key]=i.split('|')

        except:

            print('try angain:',i)

            letc.append(i)

    print('=============================================')

for key,value in skills_req.items():

    print(len(value))
#
skills_req