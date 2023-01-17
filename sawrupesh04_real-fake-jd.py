import numpy as np

import pandas as pd

import pandas_profiling as pp



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud



import missingno as ms



import string



# NLTK

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# spaCy

import spacy

from spacy import displacy

from spacy.lang.en.stop_words import STOP_WORDS
# Set Plot style

plt.style.use('fivethirtyeight')
# Load data

df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv', index_col=0)
# print head of data

df.head()
# Info of DataFrame

df.info()
df.dtypes.value_counts()
# Statistical Description

df.describe()
ms.matrix(df)

plt.show()
ms.bar(df)

plt.show()
# counts of missing value for each feature and target

df.isnull().sum()
# Drop salary_range

del df['salary_range']
# Fill null value

df.fillna("", inplace=True)
fig, ax = plt.subplots(1, 2)



sns.countplot(x='fraudulent', data=df, ax=ax[0])

ax[1].pie(df['fraudulent'].value_counts(), labels=['Real Post', 'Fake Post'], autopct='%1.1f%%')



fig.suptitle('Bar & Pie charts of Fraudulent value count', fontsize=16)

plt.show()
fig, ax = plt.subplots(1, 2)



chart = sns.countplot(x = 'required_experience', data=df[df['fraudulent']==0], ax=ax[0])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax[0].set_title('Real required experience')



chart = sns.countplot(x = 'required_experience', data=df[df['fraudulent']==1], ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax[1].set_title('Fake required experience')

plt.show()
# Create features from text columns

text_features = df[["title", "company_profile", "description", "requirements", "benefits","fraudulent"]]
# print samples of the text_features

text_features.sample(5)
columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in columns:

    text_features[col+'_len'] = text_features[col].apply(len)
real = text_features[text_features['fraudulent']==0]

fake = text_features[text_features['fraudulent']==1]



fig, ax = plt.subplots(5, 2, figsize=(15, 15))

ax[0, 0].set_title('Character Count of Real Post ')

ax[0, 1].set_title('Character Count of Fake Post ')



for i in range(5):

    for j in range(2):

        if j==0:

            ax[i, j].hist(real[columns[i]+'_len'], color='g', bins=15);

            ax[i, j].set_ylabel( columns[i] )

        else:

            ax[i, j].hist(fake[columns[i]+'_len'], color='r', bins=15);



plt.show()
for col in columns:

    text_features[col+'_len_word'] = text_features[col].apply(lambda x: len(x.split()))
real = text_features[text_features['fraudulent']==0]

fake = text_features[text_features['fraudulent']==1]



fig, ax = plt.subplots(5, 2, figsize=(15, 15))

ax[0, 0].set_title('Word Count of Real Post ')

ax[0, 1].set_title('Word Count of Fake Post ')



for i in range(5):

    for j in range(2):

        if j==0:

            ax[i, j].hist(real[columns[i]+'_len_word'], color='g', bins=15);

            ax[i, j].set_ylabel( columns[i] )

        else:

            ax[i, j].hist(fake[columns[i]+'_len_word'], color='r', bins=15);



plt.show()
def avg_word_ln(string):

    words = string.split()

    word_len = [len(word) for word in words]

    try:

        return sum(word_len)/len(words)

    except:

        return 0



for col in columns:

    text_features[col+'_avg_word_ln'] = text_features[col].apply(avg_word_ln)
real = text_features[text_features['fraudulent']==0]

fake = text_features[text_features['fraudulent']==1]



fig, ax = plt.subplots(5, 2, figsize=(15, 15))

ax[0, 0].set_title('Average Word Count of Real Post ')

ax[0, 1].set_title('Average Word Count of Fake Post ')



for i in range(5):

    for j in range(2):

        if j==0:

            ax[i, j].hist(real[columns[i]+'_avg_word_ln'], color='g', bins=15);

            ax[i, j].set_ylabel( columns[i] )

        else:

            ax[i, j].hist(fake[columns[i]+'_avg_word_ln'], color='r', bins=15);



plt.show()
# delete text_features

del text_features
# Create new feature jd (job description)

df['jd'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 
# drop features

del df['title']

del df['location']

del df['department']

del df['company_profile']

del df['description']

del df['requirements']

del df['benefits']

del df['employment_type']

del df['required_experience']

del df['required_education']

del df['industry']

del df['function']
df.head()
# Load spacy large model

nlp = spacy.load('en_core_web_lg')
df['jd'] = df['jd'].apply(str.lower)
df['jd'].iloc[0]
def remove_punctuation_and_stop_words(s):

    punctuations = list(string.punctuation)

    

    strings = " ".join([token for token in word_tokenize(s) if not token in punctuations+list(STOP_WORDS)])

    return strings

    
# Apply above function to the jd feature

df['jd'] = df['jd'].apply(remove_punctuation_and_stop_words)
# After removing puctuations and stopwords

df['jd'].iloc[0]
doc = nlp(df['jd'].iloc[0])
def lemmatization(s):

    doc = nlp(s)

    return " ".join([token.lemma_ for token in doc])
# Apply above function to the jd feature

df['jd'] = df['jd'].apply(lemmatization)
df['jd'].iloc[0]
# take first record and visualize NER

doc = nlp(df['jd'].iloc[0])



displacy.render(doc, style="ent")
displacy.render(doc, style="dep")
# WordCloud Real/Fake post



real = df[df['fraudulent']==0]['jd']

fake = df[df['fraudulent']==1]['jd']
# Real WordCloud



plt.figure(figsize = (20,20))

wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(real))

plt.imshow(wc , interpolation = 'bilinear')

plt.show()
# Fake WordCloud



plt.figure(figsize = (20,20))

wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(fake))

plt.imshow(wc , interpolation = 'bilinear')

plt.show()