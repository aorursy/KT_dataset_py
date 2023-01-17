import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file0 = '/kaggle/input/uncover/UNCOVER/world_bank/total-covid-19-tests-performed-by-country.csv'
file1 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/per-million-people-tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'
file2 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/total-covid-19-tests-performed-per-million-people.csv'
file3 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'
file4 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/total-covid-19-tests-performed-by-country.csv'

df = pd.read_csv(file0)
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df.head(3)
df4.head(3)
df = df.dropna()
fig, ax = plt.subplots(figsize=(20,6))
df[['entity', 'total_covid_19_tests']].groupby(by='entity').sum().sort_values(by='total_covid_19_tests', 
        ascending=False).head(20).plot(kind='bar', ax=ax)
plt.title('Covid Testing by Country - source:' + file0)
plt.xlabel('Country')
plt.ylabel('Number of tests performed')
df4.head(3)
fig, ax = plt.subplots(figsize=(20,6))

df4[['entity', 'total_covid_19_tests']].groupby('entity').sum().sort_values(by='total_covid_19_tests', 
                        ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file4)
plt.xlabel('Country')
plt.ylabel('Number of tests performed')
df1.head(3)
f1 = df1['code'].isna()
df1 = df1.loc[~f1, :]
fig, ax = plt.subplots(figsize=(20,4))

cols = ['entity', 'total_covid_19_tests_per_million_people', 
     'total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']

data = df1[cols].groupby('entity').sum()
data.loc[data[cols[1]]>0,:].sort_values(by=cols[2], ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file1)
plt.xlabel('Country')
plt.ylabel('Number of tests performed and cases per million people')
df2.head(3)
f1 = df2['code'].isna()
df2.loc[f1, :]
fig, ax = plt.subplots(figsize=(20,4))
cols = ['entity', 'total_covid_19_tests_per_million_people']
df2[cols].groupby('entity').sum().sort_values(by=cols[1], ascending=False).head(20).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file2)
plt.xlabel('Country')
plt.ylabel('Number of tests performed per million people')
df3.head(3)
f1 = df3['code'].isna()
df3 = df3.loc[~f1, :]
cols = ['entity', 'total_covid_19_tests','total_confirmed_cases_of_covid_19_cases']

fig,ax = plt.subplots(figsize=(20,4))
data = df3[cols].groupby('entity').sum()
data.loc[data['total_covid_19_tests']>0,:].sort_values(by=cols[2], 
                                                      ascending=False).head(30).plot(kind='bar', ax=ax)

plt.title('Covid Testing by Country - source:' + file3)
plt.xlabel('Country')
plt.ylabel('Number of tests and cases')