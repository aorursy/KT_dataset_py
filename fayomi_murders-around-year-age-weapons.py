# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')
df.drop(['Record ID','Agency Code','Agency Name','Agency Type','Incident','Victim Ethnicity','Perpetrator Ethnicity', 'Perpetrator Count','Record Source'], axis=1, inplace=True)
df.columns
df['Crime Type'].value_counts()
df=df[df['Crime Type']=='Murder or Manslaughter']
df['Crime Type'].value_counts()
print('The Year with the most murders was', df['Year'].value_counts().index[0], ' with a number:',df['Year'].value_counts().max())
print('The year with the least number was',df['Year'].value_counts().index[34], 'with the number:', df['Year'].value_counts().min())
sns.distplot(df['Year'], kde=False)
df['State'].unique()
print('The state with the most murders was', df['State'].value_counts().index[0],'with a number of:',df['State'].value_counts().max())
print('The state with the least murders was', df['State'].value_counts().index[50],'with a number of:',df['State'].value_counts().min())
#plot descending bar graph to show this

#sns.countplot(x=df['State'],data=df)

#clean victim max age and say max mean and average (then dictplot)

df['Victim Age'].max()
df=df[df['Victim Age']<90]
print('Males were',df['Victim Sex'].value_counts(normalize=True).apply(lambda x: x*100)['Male'],'percent of the victims\n'

      'Females were', df['Victim Sex'].value_counts(normalize=True).apply(lambda x: x*100)['Female'], 'percent of the victims\n'

      'The rest are unknown')
print('The average age of the victims was', df['Victim Age'].median())
plt.figure(figsize=(12,5))

sns.distplot(df['Victim Age'], kde=False)
df['Perpetrator Age'] = pd.to_numeric(df['Perpetrator Age'], errors='coerce')
df=df[(df['Perpetrator Age']<90) & (df['Perpetrator Age']>0)]
df['Perpetrator Age'].max()
plt.figure(figsize=(12,5))

sns.distplot(df['Perpetrator Age'])
#To find the most occuring perp age

df['Perpetrator Age'].mode()
#scatter plot of vic and perp ages

#sns.jointplot(x='Victim Age',y='Perpetrator Age',data=df, kind='reg')
df['Weapon'].value_counts().max()
#The list of popular weapons used

df['Weapon'].value_counts()
print('The most popular weapon used was the:',df['Weapon'].value_counts().index[0],'\naccounting for about:',df['Weapon'].value_counts().max(),'occurences')
#weapon and perp gender

plt.figure(figsize=(12, 5))

g=sns.countplot(x='Weapon', data=df, order=df['Weapon'].value_counts().index,hue='Perpetrator Sex')

for item in g.get_xticklabels():

    item.set_rotation(90)
#weapon and perp gender

plt.figure(figsize=(12,5))

r=sns.countplot(x='Weapon', data=df, order=df['Weapon'].value_counts().index,hue='Perpetrator Race')

for item in r.get_xticklabels():

    item.set_rotation(90)
plt.figure(figsize=(14,5))

w=sns.countplot(x='Relationship',data=df, order=df['Relationship'].value_counts().index[:11], hue='Weapon')

for item in w.get_xticklabels():

    item.set_rotation(90)

    