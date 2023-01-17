import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

import nltk as nlp

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/data.csv")
df.head()
df.drop(["Id","CreationDate"],axis=1,inplace=True)
df.head()
df.tail()
df.rename(columns={'Title':'title',"Body":"body","Tags":"tags","Y":"y"}, inplace=True)
df.isna().sum()
df.info()
title_list=[]

for title in df.title:

    title=re.sub("[^a-zA-Z]"," ",title)

    title=title.lower()

    title=nltk.word_tokenize(title)

    lemma  = nlp.WordNetLemmatizer()

    title=[lemma.lemmatize(word) for word in title]

    title=" ".join(title)

    title_list.append(title)
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =250

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(title_list).toarray()

print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
df_title_list = pd.DataFrame(title_list, columns = ['title'])
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df_title_list.title)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=250, background_color="white").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
body_list=[]

for body in df.body:

    body=re.sub("[^a-zA-Z]"," ",body)

    body=body.lower()

    body=nltk.word_tokenize(body)

    lemma  = nlp.WordNetLemmatizer()

    body=[lemma.lemmatize(word) for word in body]

    body=" ".join(body)

    body_list.append(body)
df_body_list = pd.DataFrame(body_list, columns = ['body'])
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =250

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(body_list).toarray()

print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df_body_list.body)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=250, background_color="white").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
tags_list=[]

for tags in df.tags:

    tags=re.sub("[^a-zA-Z]"," ",tags)

    tags=tags.lower()

    tags=nltk.word_tokenize(tags)

    lemma  = nlp.WordNetLemmatizer()

    tags=[lemma.lemmatize(word) for word in tags]

    tags=" ".join(tags)

    tags_list.append(tags)
df_tags_list = pd.DataFrame(tags_list, columns = ['tags'])
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =250

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(tags_list).toarray()

print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df_tags_list.tags)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=250, background_color="white").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
sns.barplot(x=df['y'].value_counts().index,y=df['y'].value_counts().values)

plt.title('y other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()