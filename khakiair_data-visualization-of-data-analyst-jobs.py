import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
df.info()
df['Salary Estimate'] = df['Salary Estimate'].str.strip(' (Glassdoor est.)')
df['min_Salary'] = df['Salary Estimate'].str[1:3]



df['Salary_len'] = df['Salary Estimate'].str.len()



df.loc[df['Salary_len']==10, 'max_Salary'] = df['Salary Estimate'].str[6:9]

df.loc[df['Salary_len']==9, 'max_Salary'] = df['Salary Estimate'].str[6:8]



df['min_Salary'] = df['min_Salary'].fillna(0).astype(int)

df['max_Salary'] = df['max_Salary'].fillna(0).astype(int)

df['Salary_avg'] = (df['max_Salary'] + df['min_Salary'])/2
df = df.drop('Salary_len', axis=1)
df['Company Name'] = df['Company Name'].str.replace(r'\n.*','')
df.head()
plt.subplots(figsize=(10,6))

sns.distplot(df['Salary_avg'])

plt.title('Salary Distribution', fontsize=16)

plt.show()
print(df['Size'].unique())
df.loc[df['Size']=='-1', 'Size'] = 'Unknown'



size_order = [

              '1 to 50 employees',

              '51 to 200 employees',

              '201 to 500 employees',

              '501 to 1000 employees',

              '1001 to 5000 employees',

              '5001 to 10000 employees',

              '10000+ employees',

              'Unknown'

              ]



plt.subplots(figsize=(10,6))

sns.countplot(x='Size', data=df, order=size_order)

plt.title('Company Size Distribution', fontsize=16)

plt.xlabel('Size')

plt.ylabel('Count')

plt.xticks(rotation=90)

plt.show()
df.loc[df['Sector']=='-1', 'Sector'] = 'Unknown'



plt.subplots(figsize=(12,4))

df['Sector'].value_counts().plot(kind='bar')

plt.title('Sector Distribution',fontsize=16)

plt.xlabel('Sector')

plt.ylabel('Count')

plt.show()
df.loc[df['Industry']=='-1','Industry'] = 'Unknown'



plt.subplots(figsize=(18,6))

df['Industry'].value_counts().plot(kind='bar')

plt.title('Industry Distribution',fontsize=16)

plt.xlabel('Industry')

plt.ylabel('Count')

plt.show()
plt.subplots(figsize=(18,6))

sns.countplot(x='Sector',hue='Easy Apply',data=df)

plt.title('Easy Apply by Sector',fontsize=16)

plt.xlabel('Sector')

plt.xticks(rotation=90)

plt.ylabel('Count')

plt.show()
fig = px.scatter(df, x='Rating', y='Company Name')

fig.show()
def plotAvgSalary(df, cat):

    a = df.groupby([cat])['Salary_avg'].mean()

    a = a.reset_index()

    a = a.sort_values('Salary_avg', ascending=False)

    a = a.reset_index(drop=True)

    fig = px.bar(a, x=cat, y='Salary_avg')

    fig.show()
plotAvgSalary(df, 'Company Name')
plotAvgSalary(df, 'Location')
plotAvgSalary(df,'Sector')
sample = df.groupby('Company Name').count()

sample = sample['Job Title']

sample = sample.sort_values(ascending=False)

sample = sample.reset_index()

sample = sample.head(10)

sample.rename(columns={'Job Title':'Count'})

fig = px.bar(sample, x='Company Name',y='Job Title')

fig.show()
stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='#c8d6e5',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(df['Job Title']))



plt.subplots(figsize=(10,5))

plt.axis('off')

plt.imshow(wordcloud)
stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='#f2b0a5',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(df['Job Description']))



plt.subplots(figsize=(10,5))

plt.axis('off')

plt.imshow(wordcloud)