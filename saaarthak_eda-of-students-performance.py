import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
a="../input/students-performance-in-exams/StudentsPerformance.csv"

bd=pd.read_csv(a)

bds=bd.copy()
bd.apply(lambda x: sum(x.isnull()),axis=0)
bd.describe()
bd.shape
bd["race/ethnicity"].value_counts()
bds.columns=['gender','race','ped','lunch','tpc','m','r','w']
bds.sample(3)
bds.gender.value_counts().plot(kind='bar')
bds.gender.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
bds.race.value_counts().plot(kind='bar')
bds.race.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
bds["ped"].value_counts()
bds.ped.value_counts().plot(kind='bar')
bds.ped.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()




bds["lunch"].value_counts()

bds.lunch.value_counts().plot(kind='bar')
bds.lunch.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
bds["tpc"].value_counts()
bds.tpc.value_counts().plot(kind='bar')
bds.tpc.value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
plt.rcParams['figure.figsize']=(20,10)

sns.countplot(bd['math score'])

plt.show()
sns.violinplot(y='math score',data=bd)

plt.show()
plt.rcParams['figure.figsize']=(20,10)

sns.countplot(bd['writing score'])

plt.show()
sns.violinplot(y='writing score',data=bd)

plt.show()
plt.rcParams['figure.figsize']=(20,10)

sns.countplot(bd['reading score'])

plt.show()
sns.violinplot(y='reading score',data=bd)

plt.show()
bd.mean().plot.bar()

plt.show()
bd.groupby(["test preparation course"]).mean().plot.bar()

plt.show()
bd.groupby(["parental level of education"]).mean().plot.bar()

plt.show()

bd.groupby(["gender"]).mean().plot.bar()

plt.show()
bd.groupby(["race/ethnicity"]).mean().plot.bar()

plt.show()
bd.groupby(["lunch"]).mean().plot.bar()

plt.show()
sns.countplot(x='gender',hue='tpc',data=bds)

plt.show()
sns.countplot(x='race',hue='gender',data=bds)

plt.show()
sns.countplot(x='gender',hue='ped',data=bds)

plt.show()
sns.countplot(x='gender',hue='lunch',data=bds)

plt.show()
sns.countplot(x='race',hue='tpc',data=bds)

plt.show()
sns.countplot(x='race',hue='ped',data=bds)

plt.show()
bd.corr()
bds.replace(to_replace='male', value=0, inplace=True)

bds.replace(to_replace='female', value=1, inplace=True)

bds.replace(to_replace=['group A', "group B", "group C", "group D", "group E"], value=[1,2,3,4,5], inplace=True)

bds.replace(to_replace=["bachelor's degree", 'some college', "master's degree", 

                       "associate's degree", 'high school', 'some high school'],

                        value=[5,3,6,4,2,1], inplace=True)

bds.replace(to_replace=['standard', 'free/reduced'], value=[1,2], inplace=True)

bds.replace(to_replace=['none', 'completed'], value=[0,1], inplace=True)
bds.corr()
sns.heatmap(bds.corr(),cmap="Greens")
bd['total']=bd['math score']+bd['reading score']+bd['writing score']
bd['percentage']=bd['total']/300*100
def grd(score):

    if score>=90 and score<=100:

        return 'A'

    elif score>=80 and score<90:

        return 'B'

    elif score>=70 and score<80:

        return 'C'

    elif score>=60 and score<70:

        return 'D'

    elif score>=50 and score<60:

        return 'E'

    elif  score<50:

        return 'F'

bd['grades']=bd['percentage'].apply(grd)

    
bd.sample(3)
bd.grades.value_counts().plot(kind='bar')
sns.countplot(hue='gender',x='grades',data=bd)
sns.countplot(hue='test preparation course',x='grades',data=bd)
sns.countplot(hue='lunch',x='grades',data=bd)
sns.countplot(hue='race/ethnicity',x='grades',data=bd)
sns.countplot(hue='parental level of education',x='grades',data=bd)