import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.plotly as py
df = pd.read_csv('../input/h1b_kaggle.csv')
df=df.dropna()
df['CASE_STATUS'].unique()
df.EMPLOYER_NAME.value_counts().sort_values(ascending=False).head(5)
# H1B Application trend for Top Employers in last 5 years

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(5,5))

df2=df[df['EMPLOYER_NAME'].isin(['INFOSYS LIMITED','TATA CONSULTANCY SERVICES LIMITED','WIPRO LIMITED',

                                 'DELOITTE CONSULTING LLP','IBM INDIA PRIVATE LIMITED'])]



df3=df2.groupby(['EMPLOYER_NAME','YEAR']).size().unstack()



df3.plot(kind='barh')
# Top 10 Employers with highest H1B visa applications in 2016

df1=df[df['YEAR']==2016]

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(10,10))

df1.groupby(['EMPLOYER_NAME']).size().sort_values(ascending=False).head(10).plot(kind='bar')
# H1-B Visa by Years

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(10,3))

plt.title('PETITIONS DISTRIBUTION BY YEAR')

sns.countplot(df['YEAR'])
# H1-B Visa by Status

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(10,3))

df4=df[df['CASE_STATUS'].isin(['CERTIFIED','DENIED'])]

df4.groupby(['CASE_STATUS']).size().plot(kind='barh')
# Top 25 Job Titles

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(10,10))

df.JOB_TITLE.value_counts().sort_values(ascending=False).head(25).plot(kind='barh',color='violet')
# Top 10 cities as Worksites for H1-B Visa holders

df['WORKSITE'].value_counts().head(10).plot(kind='barh')