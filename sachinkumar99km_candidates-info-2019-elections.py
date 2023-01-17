# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import os

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lets read the data 

data=pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

data.head()

# converting column names

data.columns=data.columns.str.strip().str.lower().str.replace('\n','_')

data.columns

# Lets find missing values in dataset

data.isnull().sum()

data.shape
# Lets analyse the age of candidates

data['age'].describe()

sns.boxplot(data=data['age'],y=data['age'])

plt.show()
# Let's analyze education qualification of candidates

data['education'][data['education']=='Post Graduate\n']='Post Graduate'



# Assuming NAs as Illiterate

data['education'].fillna('Illiterate',inplace=True)

data['education'][data['education']=='Not Available']='Illiterate'

data.groupby('education',as_index=False)['name'].count()
education=data.groupby('education',as_index=False)['name'].count()

education=education.sort_values(by='name',ascending=False)

sns.barplot(y=education['education'],x=education['name'])

plt.xlabel('Number of candidates')

plt.ylabel('Education qualification')

plt.show()
# Lets focus on candidates who won 

won_mps=data[data['winner']==1]

won_mps.shape
# Top 15 MP's with criminal cases

won_mps.criminal_cases=won_mps['criminal_cases'].astype(int)
crim_mps=won_mps[won_mps.criminal_cases>=1]

top_crims=crim_mps.sort_values(['criminal_cases'],ascending=False)[:15]

sns.barplot(y=top_crims['name'],x=top_crims['criminal_cases'])

plt.xlabel('Number of criminal cases')

plt.ylabel('Name of elected MPs')

plt.show()
# Karnataka MPs

won_mps_kar=won_mps[won_mps['state']=='Karnataka']
#Top 10 Karnataka MPs who bagged highest percentage of votes in their constituency

won_mps_kar[['name','constituency','over total votes polled _in constituency']].sort_values(by='over total votes polled _in constituency',ascending=False)[:10]
# Top 5 oldest MPs in Karnataka

won_mps_kar[['name','age','constituency']].sort_values(by='age',ascending=False)[:5]

# Top 20 youngest MPs across India

top25=won_mps.sort_values(by='age',ascending=True)[:20]

top25[['state','name','age','party','gender']]
# Top 5 indian political parties as per 2019 elections

party_wise_win=won_mps.groupby('party',as_index=False)['name'].count().sort_values('name',ascending=False)[:5]

sns.barplot(x=party_wise_win['name'],y=party_wise_win['party'])

plt.xlabel('Candidates Won')

plt.ylabel('Party')

plt.show()