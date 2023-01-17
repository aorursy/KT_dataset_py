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
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
df['average'] = (df['math score']+df['reading score']+df['writing score'])/3

df
df.describe()
df.info()
df.isna().sum()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
sns.distplot(df['average'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['math score'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['reading score'],hist_kws=dict(edgecolor="k", linewidth=1))
sns.distplot(df['writing score'],hist_kws=dict(edgecolor="k", linewidth=1))
df['parental level of education'].value_counts().head(30).plot(kind='barh', figsize=(10,10))
df['race/ethnicity'].value_counts().head(30).plot(kind='barh', figsize=(10,10))
df.groupby(['race/ethnicity','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(10,10))

plt.show()
new = df[['math score','reading score','writing score']].copy()



math_df = new['math score'].sum()

reading_df = new['reading score'].sum()

writing_df = new['writing score'].sum()



total = [math_df,reading_df,writing_df]

columns = ['Math','Reading','Writing']



fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(total, labels=columns, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show() 
df.groupby(['reading score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
df.groupby(['writing score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
df.groupby(['math score','gender']).size().unstack().plot(kind='bar',stacked=True,figsize=(15,15))

plt.show()
plt.figure(figsize=(10,10))

df.groupby(["test preparation course"]).mean().plot.bar(figsize=(10,10))

plt.show()
df.groupby(["parental level of education"]).mean().plot.bar(figsize=(10,10))

plt.show()


df.groupby(["race/ethnicity"]).mean().plot.bar(figsize=(10,10))

plt.show()
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

bplot = sns.boxplot( y = 'average' ,x ='parental level of education'  ,data = df)

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
fig_dims = (10,10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.boxplot(x="parental level of education", y="average", hue="gender", data=df, palette="Set1")

#sns.plt.show()