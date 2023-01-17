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
df=pd.read_csv("../input/data-police-shootings/fatal-police-shootings-data.csv")
df.head()
df.info()
df.isnull().sum()
df.describe(include='all')
df['age'].fillna(df['age'].median(),inplace=True)
df.isnull().sum()
df['race'].mode()
for column in ['armed','race', 'gender', 'flee']:

    df[column].fillna(df[column].mode()[0], inplace=True)
df.isnull().sum()
import seaborn as sns

sns.boxplot(df['age'])
sns.countplot(df['gender'])
sex=df['gender'].value_counts()

sex
m_perc=(sex[0]*100)/(sex[0]+sex[1])

f_perc=(sex[1]*100)/(sex[0]+sex[1])

print("The percentage of males and females shot out were:",m_perc,"and",f_perc)
df[df['age']==df['age'].max()]['gender']
df[df['age']==df['age'].min()]['gender']
df['armed'].value_counts(normalize=True)
sns.countplot(df['race'])
df['manner_of_death'].value_counts()
sns.countplot(df['manner_of_death'])
df['flee'].value_counts()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))

sns.countplot(hue=df['armed'],y=df['flee'])

plt.show()
df['city'].value_counts()
df['state'].value_counts()
states=df.groupby('state')

states['city'].value_counts()
sns.countplot(df['signs_of_mental_illness'],hue=df['manner_of_death'])
df['signs_of_mental_illness'].value_counts()
sns.countplot(y=df['threat_level'],hue=df['armed'])

df['threat_level'].value_counts()
sns.countplot(df['body_camera'])
y=df['date'].value_counts()
df['date']=pd.to_datetime(df['date'])

df.info()
df.groupby(df['date'].dt.year)['id'].count().plot(kind="bar")