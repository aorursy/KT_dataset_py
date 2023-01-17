import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/StudentsPerformance.csv")
print(df.head())
print('Gender')

print(df.gender.value_counts())
print('Parental Education')

print(df['parental level of education'].value_counts())

labels= 'Some college','associatesdegree','high school','some high school','bachelor degree', 'masterdegree'

sizes=[226,222,196,179,118,59]

colors=['red','green','blue','pink','yellow','orange']

plt.pie(sizes,labels=labels,colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
ax=sns.countplot(x='parental level of education', data = df, hue='gender')

ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)

plt.show()

sns.countplot(x=df['lunch'])

plt.title('Lunch type analysis')

plt.show()



sns.countplot(x=df['test preparation course'])

plt.title('Test Preperation analysis')

plt.show()
sns.boxplot(x='gender',y='writing score',data=df,palette='viridis')

plt.show()
sns.boxplot(x='gender',y='math score',data=df,palette='viridis')

plt.show()
sns.boxplot(x='gender',y='reading score',data=df,palette='viridis')

plt.show()
plt.figure(figsize=(10,5))

sns.stripplot(x='parental level of education',y='writing score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='test preparation course',y='writing score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='lunch',y='writing score',data=df,palette='cividis')

plt.show()
plt.figure(figsize=(10,5))

sns.stripplot(x='parental level of education',y='math score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='test preparation course',y='math score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='lunch',y='math score',data=df,palette='cividis')

plt.show()
plt.figure(figsize=(10,5))

sns.stripplot(x='parental level of education',y='reading score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='test preparation course',y='reading score',data=df,palette='cividis')

plt.show()



plt.figure(figsize=(10,5))

sns.stripplot(x='lunch',y='reading score',data=df,palette='cividis')

plt.show()
ax = sns.countplot(x="math score", data = df,palette="muted")

ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

plt.show()



ax = sns.countplot(x="reading score", data = df,palette="muted")

ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

plt.show()



ax = sns.countplot(x="writing score", data = df,palette="muted")

ax.set_xticklabels(ax.get_xticklabels(), fontsize=6)

plt.show()
avg_math=df['math score'].mean()

print("On average mark scored in math subject is :{:.2f}%".format(avg_math))



avg_read=df['reading score'].mean()

print("On average mark scored in reading subject is :{:.2f}%".format(avg_read))



avg_write=df['writing score'].mean()

print("On average mark scored in writing subject is :{:.2f}%".format(avg_write))

passmark=40

df['Math_SubjectStatus'] = np.where(df['math score']<passmark, 'F', 'P')

df['Reading_SubjectStatus'] = np.where(df['reading score']<passmark, 'F', 'P')

df['Writing_SubjectStatus'] = np.where(df['writing score']<passmark, 'F', 'P')

                                    

df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_SubjectStatus'] == 'F' or 

                                    x['Reading_SubjectStatus'] == 'F' or x['Writing_SubjectStatus'] == 'F' else 'P', axis =1)



print(df.OverAll_PassStatus.value_counts())



p = sns.countplot(x='parental level of education', data = df, hue='OverAll_PassStatus', palette='bright')

plt.show()

p = sns.countplot(x='lunch', data = df, hue='OverAll_PassStatus', palette='bright')

plt.show()

p = sns.countplot(x='test preparation course', data = df, hue='OverAll_PassStatus', palette='bright')

plt.show()
df['Total_Marks'] = df['math score']+df['reading score']+df['writing score']

df['Percentage'] = df['Total_Marks']/3

def GetGrade(Percentage, OverAll_PassStatus):

    if ( OverAll_PassStatus == 'F'):

        return 'F'    

    if ( Percentage >= 80 ):

        return 'A'

    if ( Percentage >= 70):

        return 'B'

    if ( Percentage >= 60):

        return 'C'

    if ( Percentage >= 50):

        return 'D'

    if ( Percentage >= 40):

        return 'E'

    else: 

        return 'F'



df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

print(df.Grade.value_counts())



sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")

plt.show()

ad = sns.countplot(x='parental level of education', data = df, hue='Grade', palette='bright')

ad.set_xticklabels(ax.get_xticklabels(), fontsize=7)

plt.show()



p = sns.countplot(x='test preparation course', data = df, hue='Grade', palette='bright')

plt.show()



p = sns.countplot(x='lunch', data = df, hue='Grade', palette='bright')

plt.show()