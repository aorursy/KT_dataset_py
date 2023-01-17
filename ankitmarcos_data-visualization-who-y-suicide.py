#Please run this cell in Google Co-Lab



!wget https://www.dropbox.com/s/f3pey8rb5elq4ua/who_suicide_statistics.csv
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
df = pd.read_csv('who_suicide_statistics.csv') #read_csv function is used to read the dataset into a pandas dataframe



df.head() #Prints the first 5 rows of the data
print('Number Of Rows:',df.shape[0])

print('Number Of Columns:', df.shape[1])
df.nunique()
a = df.columns

for i in range(0, len(a)):

  print('Name of Column',i+1,':',a[i])
df.isnull().sum()
df = df.dropna() #Dropping Rows if the row contains all NaN Values



df.isnull().sum()
df.info()
df.describe()
sns.set()

plt.figure(figsize=(12,8))

sns.distplot(df['year'], kde = False)



plt.xlabel('Year')

plt.ylabel('Count')



plt.show() 
fig = px.histogram(df, x="year")

fig.show()
plt.figure(figsize = (12,8))

ax = sns.countplot(df["year"], color='green')



plt.xticks( rotation=90 )



plt.show()
df.head()
df['age'].nunique()
a = df['age'].unique()
lt = [20, 29, 45, 8, 60, 80]



for i in range(0, len(a)):



  df.replace(to_replace = a[i], value = lt[i], inplace=True) 

df['age'].unique()
df.head()
plt.figure(figsize=(12, 8))



sns.barplot(df['sex'],df['suicides_no'])



plt.xlabel('Sex: Male or Female')

plt.ylabel('No. of Suicides')



plt.show()
plt.figure(figsize=(12,8))



fig = px.bar(df, x= 'year', y='suicides_no' )



fig.show()
df.head()
fig = px.scatter(x= df['year'], y=df['population'])



fig.show()