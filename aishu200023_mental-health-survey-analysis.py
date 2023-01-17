# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/mental-health-in-tech-survey/survey.csv')
df.info()
df.shape
df.isnull().sum()
df.drop(columns=['state','comments'],inplace=True)
df['self_employed'].fillna('No',inplace=True)

df['work_interfere'].fillna('Sometimes',inplace=True)
df.isnull().sum()
df.columns
df.duplicated().sum()
df.drop(df[df['Age'] < 0].index, inplace = True) 

df.drop(df[df['Age'] > 100].index, inplace = True) 
sns.boxplot(df['Age'])
df['Gender'].unique()
#will decrease the number of categoried in Gender

df['Gender'] = df['Gender'].str.lower()

male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]

trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]

female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

df['Gender'] = df['Gender'].apply(lambda x:"Male" if x in male else x)

df['Gender'] = df['Gender'].apply(lambda x:"Female" if x in female else x)

df['Gender'] = df['Gender'].apply(lambda x:"Trans" if x in trans else x)

df.drop(df[df.Gender == 'p'].index, inplace=True)

df.drop(df[df.Gender == 'a little about you'].index, inplace=True)
df['Gender'].unique()
sns.countplot(df['Gender'])
#country- wise gender ratio participating in the survey

#shows that more number of males are working in tech companies all over the world

plt.figure(figsize= (20,9))

sns.countplot(x='Country', order= df['Country'].value_counts().index, hue='Gender', data=df)

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=90)
sns.barplot(df['mental_health_consequence'].unique(),df['mental_health_consequence'].value_counts())
#country wise representation of data with focus on India

plt.figure(figsize=(20,8))

sns.countplot(df.Country, order= df['Country'].value_counts().index)

plt.xticks(rotation=90)

plt.annotate('Mental Health Survey Participants from India', xy=(8, 20), xytext=(10, 20.5),

             arrowprops=dict(facecolor='black', shrink=0.05),)
plt.pie(df['coworkers'].value_counts(),labels=df['coworkers'].unique())

df['coworkers'].value_counts()
#So people dont know exactly whether employer would consider mental health as serious as a physical one.Now we can analyse it

plt.hist(df['mental_vs_physical'],histtype='step')
#family history vs mental health

plt.figure(figsize=(12,8))

sns.countplot(y="family_history", hue="treatment", data=df)

plt.title("Does family hisitory effects mental health ? ",fontsize=20,fontweight="bold")

plt.ylabel("")

plt.show()

#Corelation of features

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

for i in df.columns:

    df[i] = number.fit_transform(df[i].astype('str'))
features_correlation = df.corr()

plt.figure(figsize=(8,8))

sns.heatmap(features_correlation,vmax=1,square=True,annot=False,cmap='Blues')

plt.show()