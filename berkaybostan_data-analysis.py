# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot

import cufflinks as cf

init_notebook_mode(connected = True)

cf.go_offline()



sns.set_style("darkgrid")

df = pd.read_csv("../input/data.csv")

df.head(2)
plt.figure(figsize=(16,4))

sns.countplot(df.gender,palette="Set1",hue=df.marital_status)
plt.figure(figsize=(16,4))

sns.countplot(df.disease,palette="coolwarm")

plt.xticks(rotation="90")
Disease = df['disease'].value_counts()

Disease.iplot(kind = 'bar', theme = 'pearl',colors = 'Blue', xTitle = 'Disease', yTitle = 'Count', title = 'Disease Frequency')
plt.figure(figsize=(20,8))

sns.countplot(df.gender,hue=df.disease,palette="rainbow")
plt.figure(figsize=(20,8))

sns.countplot(df.employment_status,hue=df.disease,palette="coolwarm")
plt.figure(figsize=(20,8))

sns.countplot(df.employment_status[df.employment_status=="student"],hue=df.disease,palette="coolwarm")
plt.figure(figsize=(16,8))

#sns.countplot(df.education)

Education = df['education'].value_counts()

Education.iplot(kind = 'bar', theme = 'pearl',colors = 'Blue', xTitle = 'Education', yTitle = 'Count')
plt.figure(figsize=(16,4))

sns.countplot(df.education,hue=df.employment_status,palette="rainbow")
plt.figure(figsize=(16,4))

sns.countplot(df.ancestry,palette="coolwarm")

plt.xticks(rotation=90)

#Ancestry = df['ancestry'].value_counts()

#Ancestry.iplot(kind = 'bar', theme = 'pearl',colors = 'Blue', xTitle = 'Ancestry', yTitle = 'Count')
plt.figure(figsize=(16,8))

sns.countplot(df.ancestry[df.ancestry=="Netherlands"],hue=df.disease,palette="coolwarm")
plt.figure(figsize=(16,8))

sns.countplot(df.military_service,hue=df.disease,palette="coolwarm")
plt.figure(figsize=(16,4))

sns.distplot(df.daily_internet_use)
Year_of_birth = [ ]

for str in list(df['dob']):

    year = int(str.split('-')[0])

    Year_of_birth.append(year)

df['YOB'] = Year_of_birth

df.head()



df['AGE'] = 2019 - df['YOB']

df['AGE'].value_counts().iplot(kind = 'bar', theme = 'pearl', colors = 'Blue', xTitle = 'Age', yTitle = 'count')

disease = list(df['disease'].unique())



for x in disease:

    trace = df[df['disease'] == x].groupby('gender').count()['id']

    trace.iplot(kind = 'bar', title = x, theme = 'pearl',colors = 'Blue')
disease = list(df['disease'].unique())



for x in disease:

    trace = df[df['disease'] == x].groupby('ancestry').count()['id']

    trace.iplot(kind = 'bar', title = x, theme = 'pearl',colors = 'Blue')