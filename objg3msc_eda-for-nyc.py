import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/2016 School Explorer.csv')
df.head()
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
df2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df2
duplicate_bool=df.duplicated()
duplicate=df.loc[duplicate_bool == True]
print(duplicate)
df1=df.sort_values(by=['Economic Need Index'],ascending=False)
df1=df1.iloc[:500,]
print(df1[['City','Economic Need Index']])
sns.stripplot(x="Economic Need Index", y="City",hue="Community School?",data=df1)
subset=df.loc[(df.City.isin(['NEW YORK','BROOKLYN'])& (df['Community School?']=="Yes"),['Student Attendance Rate'])]
subset
subset['count']=subset.index
sns.barplot(x="count",y="Student Attendance Rate",data=subset)

df.reset_index(inplace=True)
a=df.pivot_table(values = 'index', index =['Collaborative Teachers Rating'],columns=df['Supportive Environment %']>"70%", aggfunc = 'count')
print(a)
a.columns=['Lessthan70','Greaterthan70']
a['Collaborative Teachers Rating']=a.index
sns.barplot(x="Greaterthan70", y="Collaborative Teachers Rating",data=a,color="blue") 
sns.lmplot(x="Average ELA Proficiency", y="Average Math Proficiency", hue="Community School?", markers=["*","+"], palette="Set2", data=df)
col1=['Rigorous Instruction %']
df[col1]=df[col1].replace({'\%':' '},regex=True)
df[col1]=df[col1].astype('float64')
col2=['Effective School Leadership %']
df[col2]=df[col2].replace({'\%':' '},regex=True)
df[col2]=df[col2].astype('float64')
col3=['Collaborative Teachers %']
df[col3]=df[col3].replace({'\%':' '},regex=True)
df[col3]=df[col3].astype('float64')
col4=['Percent of Students Chronically Absent']
df[col4]=df[col4].replace({'\%':' '},regex=True)
df[col4]=df[col4].astype('float64')
col5=['Supportive Environment %']
df[col5]=df[col5].replace({'\%':' '},regex=True)
df[col5]=df[col5].astype('float64')

a=df[col1]
b=df[col2]
c=df[col3]
d=df[col4]
e=df[col5]
result=pd.concat([a, b,c,d,e], axis=1)
#result
plt.figure(1)
sns.jointplot(x="Percent of Students Chronically Absent", y="Supportive Environment %",data=result)
plt.figure(2)
sns.jointplot(y="Effective School Leadership %", x="Rigorous Instruction %",data=result)
plt.figure(3)
sns.jointplot(y="Effective School Leadership %", x="Collaborative Teachers %",data=result)
#Display the Mean ELA and Math Scores for Black/Hispanic Dominant Schools
df[df['Percent Black / Hispanic'] >= '70%'][['Average ELA Proficiency','Average Math Proficiency']].mean()
#Display the Mean ELA and Math Scores for White/Asian Dominant Schools
df[df['Percent Black / Hispanic'] <= '30%'][['Average ELA Proficiency','Average Math Proficiency']].mean()
# Create New Column for Black/Hispanic Dominant Schools
df['Black_Hispanic_Dominant'] = df['Percent Black / Hispanic'] >='70%'

#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average Math Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average Math Proficiency'] , color='r',shade=True, label='Asian/White Dominant School')
plt.title('Average Math Proficiency Distribution by Race')
plt.xlabel('Average Math Proficiency Score')
plt.ylabel('Frequency Count')
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average ELA Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average ELA Proficiency'] , color='r',shade=True, label='Asian/White Dominant School')
plt.title('Average ELA Proficiency Distribution by Race')
plt.xlabel('Average ELA Proficiency Score')
plt.ylabel('Frequency Count')