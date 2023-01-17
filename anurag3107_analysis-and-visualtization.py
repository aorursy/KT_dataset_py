import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv',index_col='sl_no') 

df_b=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv',index_col='sl_no') ##backup data for another set of analysis
df.head()
df['status'].value_counts()
a=df.corr() ## finding the correation of every column to know the relation better

a
df['workex']=df['workex'].map({'Yes':1,'No':0})

df['gender']=df['gender'].map({'M':1,'F':0})

df['status']=df['status'].map({'Placed':1,'Not Placed':0})
df.head()
a=df.corr()

plt.figure(figsize=(10,6))

sns.heatmap(a,cmap='coolwarm')

plt.title('Co-relation of every feature to see relation with placement status')

plt.show()
## we can further verify the above analysis by a boxplot of ssc percentage and placement status

plt.figure(figsize=(10,6))

sns.boxplot('ssc_p','status',data=df_b,palette='magma')

plt.title('SSC percantage spread vs Placement status')

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x='gender',data=df_b,hue='status',palette='rainbow')

plt.title('Placement status v/s Gender')

plt.show()
df['specialisation'].value_counts()
plt.figure(figsize=(10,6))

sns.countplot(x='specialisation',data=df_b,hue='status',palette='plasma_r')

plt.title('PLacement numbers according to specialisation in MBA')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='specialisation',y='salary',data=df,palette='winter')

plt.title('Salary comparison of different specialisation in MBA')

plt.show()
plt.figure(figsize=(10,6))

sns.violinplot(x='hsc_s',y='salary',data=df,palette='twilight')

plt.title('Salary comparison with respect to HSC subject')