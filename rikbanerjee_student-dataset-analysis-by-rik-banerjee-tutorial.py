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
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
df.info()
df.describe()
#show columns

for i,col in enumerate(df.columns):

    print(i+1,". column is ",col)
df["race/ethnicity"].nunique()
df["race/ethnicity"].value_counts()
sns.countplot(y="race/ethnicity",data=df,palette="magma")
plt.figure(figsize=(10,4))

sns.barplot(x=df['race/ethnicity'].value_counts().index,y=df['race/ethnicity'].

               value_counts().values,palette="Blues_d",hue=['group A','group B','group C','group D','group E'])
math=df["math score"]

reading=df["reading score"]

writing=df["writing score"]

score_list=[math,reading,writing]
for i in score_list:

    print(np.mean(i))
f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)

for i in range(len(score_list)):

    sns.distplot(score_list[i],hist=False,color="magenta",ax=axes[i])



plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
sns.kdeplot(df['math score'],bw=.15)

plt.xlabel('Math Score')

plt.ylabel('Frequency')

plt.title('Math Score Show Kde Plot')

plt.show()
sns.violinplot(df['math score'])
for i in range(len(score_list)):

    sns.distplot(score_list[i],hist=False,kde_kws = {'shade':True, 'linewidth': 3.5},label=i)

plt.ylabel("frequency")

plt.xlabel("Scores")

    
group=df.groupby(["gender","race/ethnicity"])["math score","reading score","writing score"].mean()

group
plt.figure(figsize=(10,4))

sns.boxplot(x="race/ethnicity",y="math score",hue="gender",data=df,color="cyan")

plt.title("Male vs Female math score mean Groupwise")
plt.figure(figsize=(10,4))

sns.boxplot(x="race/ethnicity",y="reading score",hue="gender",data=df,color="red")

plt.title("Male vs Female reading score mean Groupwise")
plt.figure(figsize=(10,4))

sns.boxplot(hue="race/ethnicity",y="math score",x="gender",data=df)

plt.title("Male vs Female math score mean Groupwise")
sns.violinplot(x=df['race/ethnicity'],y=df['math score'])

plt.show()
sns.violinplot(x=df["race/ethnicity"],y=df["math score"],hue=df["gender"],split=True,palette="Set2")
sns.factorplot(x="race/ethnicity", y="math score", hue="gender",col='lunch',data=df)
sns.catplot(x="race/ethnicity", y="math score", hue="gender",col='lunch',data=df)
list1=["group A","group B","group C","group D","group E"]

list2=["math score","reading score","writing score"]

f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)

for j in range(len(list2)):

    for i in list1:

        df_=df[df["race/ethnicity"]==i]

        sns.distplot(df_[list2[j]],hist=False,kde_kws = {'shade': False, 'linewidth': 1.5},label=i,ax=axes[j])



plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
#now gender wise distribution

list3=["male","female"]

list2=["math score","reading score","writing score"]

f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)

for j in range(len(list2)):

    for i in list3:

        df_=df[df["gender"]==i]

        sns.distplot(df_[list2[j]],hist=False,kde_kws = {'shade': True, 'linewidth': 3},label=i,ax=axes[j])

        

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
#now finding relation between parental education and scores



df["parental level of education"].value_counts()
list4=["some college","associate's degree","high school","some high school","bachelor's degree","master's degree"]

plt.figure(figsize=(18,5))

for i in list4:

    df_p=df[df["parental level of education"]==i]

    sns.distplot(df_p["math score"],hist=False,kde_kws = {'shade': True, 'linewidth': 1.5},label=i)
list4=["some college","associate's degree","high school","some high school","bachelor's degree","master's degree"]

list2=["math score","reading score","writing score"]

f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)

for j in range(len(list2)):

    for i in list4:

        df_p=df[df["parental level of education"]==i]

        sns.distplot(df_p[list2[j]],hist=False,kde_kws = {'shade': True, 'linewidth': 1.5},label=i,ax=axes[j])

        

plt.setp(axes, yticks=[])

plt.tight_layout()

plt.show()
sns.pairplot(df,hue="gender",palette="husl")
sns.lmplot(x='math score',y='writing score',hue='gender',data=df,markers=['+','o'])

plt.xlabel('Math Score')

plt.ylabel('Writing Score')

plt.title('Math Score vs Writing Score')

plt.show()
sns.kdeplot(df['reading score'],df['writing score'],cmap='Blues',shade=True,shade_lowest=False)
sns.kdeplot(df['reading score'],df['math score'],cmap='twilight',shade=True,shade_lowest=False)
plt.figure(figsize=(18,5))

sns.stripplot(x="math score",y="reading score",alpha=0.5,data=df,jitter=0.2)
plt.figure(figsize=(6,6))

sns.stripplot(x="gender",y="writing score",alpha=0.5,data=df,jitter=0.3)
plt.figure(figsize=(10,5))

sns.set(style="whitegrid")

sns.swarmplot(x="race/ethnicity",y="writing score",data=df,palette="Set2")
plt.figure(figsize=(10,5))

sns.set(style="whitegrid")

sns.swarmplot(x="race/ethnicity",y="math score",data=df,palette="Set2")