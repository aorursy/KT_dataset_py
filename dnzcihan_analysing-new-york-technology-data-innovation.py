import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import re 

import nltk as nlp

from plotly.offline import iplot





df = pd.read_csv('../input/nyc-jobs.csv',sep=',')

df.head()
df.info()


df=df[df['Job Category']=='Technology, Data & Innovation'].reset_index()

df = df.drop(['index'],axis=1)

df
sns.countplot(x='Full-Time/Part-Time indicator', data=df)

sns.countplot(x='Salary Frequency', data=df)
df['civil_service'] =df['Civil Service Title']





groups = df.groupby(['civil_service']).size()

plt.figure(figsize=(10, 10))

groups.plot.barh()



df['business_title']=df['Business Title']





groups = df.groupby(['business_title']).size()

plt.figure(figsize=(10, 30))

groups.plot.barh()
df['location']=df['Work Location']

groups = df.groupby(['location']).size()

plt.figure(figsize=(10, 10))

groups.plot.barh()
df['workUnit']=df['Division/Work Unit']

groups = df.groupby(['workUnit']).size()

plt.figure(figsize=(10, 15))

groups.plot.barh()
df['mean_salary_from']=df['Salary Range From']

a=df.groupby('business_title')['mean_salary_from'].mean()

plt.figure(figsize=(10, 30))

a.plot.barh()
df['skills']=df['Preferred Skills']





first_description=df.skills[4]

description=re.sub("[^a-zA-Z]"," ",first_description)



description = description.lower()

print("First value :::::::::::::: {0}   \nSecond value :::::::::  {1}".format(first_description,description))
description_list=[]

import nltk

for description in df.skills:

     description=str(description)

     description = description.lower()

     description = nltk.word_tokenize(description)

     lemma = nlp.WordNetLemmatizer()

     description = [lemma.lemmatize(word) for word in description]

     description =" ".join(description)

     description_list.append(description)

from sklearn.feature_extraction.text import CountVectorizer

max_features=60

count_vectroizer =CountVectorizer(max_features=max_features,stop_words="english")# -----> stopwords unmeaning words

sparce_matrix = count_vectroizer.fit_transform(description_list).toarray()

dictionary = count_vectroizer.vocabulary_.items()  

vocab = []

count = []

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

vocab_bef_stem = pd.Series(count, index=vocab)

vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)

top_vacab = vocab_bef_stem.head(50)

top_vacab.plot(kind = 'barh', figsize=(10,20))
description_list2=[]

df['minR']=df['Minimum Qual Requirements']

for description2 in df.minR:

     description2=str(description2)

     description2 = description2.lower()

     description2 = nltk.word_tokenize(description2)

     lemma2 = nlp.WordNetLemmatizer()

     description2 = [lemma2.lemmatize(word) for word in description2]

     description2 =" ".join(description2)

     description_list2.append(description2)

        

        

from sklearn.feature_extraction.text import CountVectorizer

max_features=60

count_vectroizer2 =CountVectorizer(max_features=max_features,stop_words="english")# -----> stopwords unmeaning words

sparce_matrix2 = count_vectroizer2.fit_transform(description_list2).toarray()

dictionary2 = count_vectroizer2.vocabulary_.items()  

vocab2 = []

count2 = []

for key, value in dictionary2:

    vocab2.append(key)

    count2.append(value)

vocab_bef_stem2 = pd.Series(count2, index=vocab2)

vocab_bef_stem2 = vocab_bef_stem2.sort_values(ascending=False)

top_vacab2 = vocab_bef_stem2.head(50)

top_vacab2.plot(kind = 'barh', figsize=(10,20))
