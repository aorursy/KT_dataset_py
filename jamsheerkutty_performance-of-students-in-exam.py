# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
SP=pd.read_csv('../input/StudentsPerformance.csv')
# check first 5 records

SP.head()
#check the size of the data

print('No of rows = {}'.format(SP.shape[0]))

print('No of columns = {}'.format(SP.shape[1]))
#checking null values

SP.isnull().sum()
#checking the datatypes

SP.info()
#Checking values of gender wise



SP['gender'].value_counts()
SP['parental level of education'].value_counts()
SP.head()
#grouping data based on race and gender

SP.groupby('gender')['race/ethnicity'].value_counts()
#grouping data based on parental level education and gender

SP.groupby('gender')['parental level of education'].value_counts()
#Lets see how many did test preparation course

SP.groupby('gender')['test preparation course'].value_counts()
# Lets plot some fun stuffs for bivarite analysis
sns.countplot(x='race/ethnicity',hue='gender',data=SP)


ax = sns.countplot(x="parental level of education",hue='gender',data=SP)



ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)

plt.tight_layout()

plt.show()

sns.countplot(x='lunch',hue='gender',data=SP)
sns.countplot(x='test preparation course',hue='gender',data=SP)
fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

sns.countplot(x = 'math score', data = SP)

ax.set_xlabel('Maths Scores', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Score distribution of maths', fontsize=15)

sns.despine()

fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

sns.countplot(x = 'reading score', data = SP)

ax.set_xlabel('Reading Scores', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Score distribution of Reading', fontsize=15)

sns.despine()

fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

sns.countplot(x = 'writing score', data = SP)

ax.set_xlabel('Writing Scores', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Score distribution of Writing', fontsize=15)

sns.despine()

SP["maths_result"] = np.where(SP["math score"] >=40, "Pass", "Fail")

SP["reading_result"]= np.where(SP["reading score"] >=40, "Pass", "Fail")

SP["writing_result"] = np.where(SP["writing score"] >=40, "Pass", "Fail")



SP["Total"]=SP["math score"] + SP["reading score"] + SP["writing score"]

SP.head(3)

#calculate percentage of each student



SP["Percentage"]=round((SP["Total"]/300)*100,2)

SP.head(5)