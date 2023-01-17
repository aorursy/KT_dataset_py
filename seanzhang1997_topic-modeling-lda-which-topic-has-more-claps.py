import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import string

import re

from wordcloud import WordCloud

import warnings

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")
top20_articles = df.sort_values(["claps"],ascending=0)[0:20][["title","claps","publication","url"]]

top20_articles.reset_index(drop=True, inplace=True)

top20_articles

# As we can see the most popular articles are mainly from The Startup. There is only one

# article from Towards Data Science. Go ahead and explore them. They are great articles.
# What are the article distribution by publications?

plt.figure(figsize=(20,5))

sns.countplot("publication", data=df, palette="bright")

# Towards Data Science rank No.2
# Average claps and responses for different publications. 

avg_by_publication = df.groupby(by="publication").mean()[["claps"]]

avg_by_publication.sort_values(["claps"],ascending=False)

# Medium bloggers should start to write articles in Better Humans if they want 

# more reactions from readers.
#Firstly lets see the wordcloud for towards data science.

nltk.download('stopwords')

df_data_science = df[df["publication"]=="Towards Data Science"]

title_data = "".join(str(x) for x in df_data_science["title"])

subtitle_data = "".join(str(x) for x in df_data_science["subtitle"])

title_data = title_data+subtitle_data

stop_words = set(nltk.corpus.stopwords.words("english"))

word_cloud = WordCloud(stopwords=stop_words, width=2000, height=1000,\

                            max_font_size=160, min_font_size=30).generate(title_data)

plt.figure(figsize=(12,6), facecolor="k")

plt.imshow(word_cloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show() #Aha, we saw a lot of familar topics.We should start to figure out these topics now!
import unicodedata

def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text



def remove_stopwords(text):

    filtered_text = []

    for i in text.split():

        i = i.strip().lower()

        if i not in stop_words:

            filtered_text.append(i)

    filtered_text = ' '.join(filtered_text)    

    return filtered_text



def normalize_accented_characters(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')

    return text



def normalize_text(text):

    text = remove_punct(text)

    text = remove_stopwords(text)

    text = normalize_accented_characters(text)

    return text    
df_data_science = df[df["publication"]=="Towards Data Science"]

df_data_science["title"] = df_data_science["title"].astype(str) + df_data_science["subtitle"].astype(str)

df_data_science.drop(columns=["subtitle","id","url","image","publication"],inplace=True)

df_data_science["title"] = df_data_science["title"].apply(normalize_text)

df_data_science
import sklearn

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.6,min_df=2)

# max_df is the maximum precentage this word show in the documents.

# I tweaked this parameter for topic modelling because some words like "Data Science"

# are too frequent in the documents.

# min_df means at least it shows min_df times in the documents.

doc_term_matrix = count_vect.fit_transform(df_data_science["title"].values)

doc_term_matrix
from sklearn.decomposition import LatentDirichletAllocation

# LDA requires us to specify the number of topics. So that will be hp to tweak.

number_topics = 3

number_words = 20

LDA = LatentDirichletAllocation(n_components=number_topics, n_jobs=-1)

LDA.fit(doc_term_matrix)

#A helper function. You can save this snippet for future use.

def print_topics(model, count_vectorizer, n_top_words):

    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        print("\nTopic #%d:" % topic_idx)

        print(" ".join([words[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("Topics:")

print_topics(LDA, count_vect, number_words)        
!pip install pyLDAvis #visualizing LDA
import pyLDAvis

import pyLDAvis.sklearn

from pyLDAvis import sklearn as sklearn_lda

#prepare to display result in the Jupyter notebook

pyLDAvis.enable_notebook()

LDAvis = sklearn_lda.prepare(LDA, doc_term_matrix, count_vect)

#run the visualization [mds is a function to use for visualizing the "distance" between topics]

LDAvis
#Allocate articles into topics.

topic_values = LDA.transform(doc_term_matrix)

df_data_science['topic'] = topic_values.argmax(axis=1)

df_data_science.head(10)

#Ok, lets humanly test the first 10 -.-

#[1,1,0,1,1,1,0,0,1,1] 7 out of 10 are correct. 
df_data_science["topic"].replace(0,"Deep Learning and AI",inplace=True)

df_data_science["topic"].replace(1,"General Data Analysis",inplace=True)

df_data_science["topic"].replace(2,"Machine Learning Modeling",inplace=True)

avg_claps_by_topic = df_data_science.groupby(by=["topic"]).mean()["claps"].reset_index()

avg_reading_time_by_topic = df_data_science.groupby(by=["topic"]).mean()["reading_time"].reset_index()
print(avg_claps_by_topic) # No significant difference.

print(avg_reading_time_by_topic) # More reasonable.