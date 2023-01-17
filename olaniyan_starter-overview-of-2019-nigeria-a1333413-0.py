# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns # Exploratory Analysis
%matplotlib inline
df = pd.read_excel('../input/2019-GENERAL-ELECTIONS-FINAL-LIST-OF-PRESIDENTIAL-CANDIDATES.xlsx')
df.info()
df.drop(['STATE OF ORIGIN', 'Twitter Handle', 'Official Website'], axis=1, inplace=True)
df.describe()
df.head(10)
# PWD column should be removed, having an input None



df.drop('PWD',axis = 1,inplace=True)
# This shows the candidate are either contesting as Vice-President or President

df['POSITION'].unique()
print('PRESIDENT: \n'+ str(df[df['POSITION'] == 'PRESIDENT'].count()))

print('\n')

print('VICE-PRESIDENT: \n'+ str(df[df['POSITION'] == 'VICE-PRESIDENT'].count()))
df[df['REMARKS'].notna()]
# Set the aesthetic style of the plots.

sns.set_style("whitegrid")
sns.countplot(x = 'GENDER', data=df)
df['GENDER'].value_counts()
f_data = df[df['GENDER'] == 'F']
f_data.head(28)
# Number of Female Vice-President Candidate

sum(df[df['GENDER'] == 'F']['POSITION'] == 'VICE-PRESIDENT')
# Number of Female Vice-President Candidate

sum(df[df['GENDER'] == 'F']['POSITION'] == 'PRESIDENT')
df[df['GENDER'] == 'F'].max()
df[df['GENDER'] == 'F'].min()
old = df['AGE'].max()

young = df['AGE'].min()

print('Oldest Candidate:'+str(old)+' yrs old')

print('Youngest Candidate:'+str(young)+' yrs old')
df.loc[df['AGE'].idxmax()]
df.loc[df['AGE'].idxmin()]
df.head(10)
df.tail(10)
no_cand = 144 # number of candidates

no_part = no_cand/2 # number of parties

print('Number of Parties Contesting are: '+str(no_part)+' Parties')
df[df['QUALIFICATION'] == 'SSCE']
df[df['QUALIFICATION'] == 'FSLC']
y = range(0,41)



def impute_age(age):

    if age in y:

        return True

    else: 

        return False
df[df['AGE'].apply(lambda x: impute_age(x))].count()