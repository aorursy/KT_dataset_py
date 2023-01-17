# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 
import seaborn as sns 
import spacy 
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,  PCA, NMF
import random 
df = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200512.csv',index_col='Unnamed: 0')

df2 = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200526.csv',index_col='Unnamed: 0')
df2.head()

df3 = pd.read_csv('/kaggle/input/covid19-public-media-dataset/covid19_articles_20200504.csv',index_col='Unnamed: 0')
df3.head()

# concatenating sources 
df = pd.concat([df,df2,df3],ignore_index=True)
df.head()
# visualize sources and their counts 
order_by = df['domain'].value_counts().index
sns.catplot(data=df,x='domain',kind='count',aspect=3,order=order_by)
plt.xticks(rotation=90)
plt.show()
# topic areas
# here we can see 7 topic_areas, where general category is in majority 
order_by = df['topic_area'].value_counts().index
sns.catplot(kind='count',x='topic_area',aspect=3,data=df,order=order_by)
plt.show()
# dates aggregation 
df.date.value_counts().sort_index().plot()
plt.xticks(rotation=90)
plt.show()
nlp = spacy.load('en_core_web_sm',disable=['parser','ner','tokenizer'])
# from https://www.kaggle.com/jannalipenkova/covid-19-media-overview
RELEVANT_POS_TAGS = ["PROPN", "VERB", "NOUN", "ADJ"]

CUSTOM_STOPWORDS = ["say", "%", "will", "new", "would", "could", "other", 
                    "tell", "see", "make", "-", "go", "come", "can", "do", 
                    "such", "give", "should", "must", "use"]

# Data Preprocessing 
def preprocess(txt):
  '''
  Take text pass through spacy's pipeline 
  Normalize text using remove stopwords from CUSTOM_STOPWORDS, take words which are RELEVENT_POS_TAGS and
  take lemma and use that in smaller version of alphabet
  '''
  doc = nlp(txt)
  rel_tokens = " ".join([tok.lemma_.lower() for tok in doc if tok.pos_ in RELEVANT_POS_TAGS and tok.lemma_.lower() not in CUSTOM_STOPWORDS])
  return rel_tokens
tqdm.pandas()
processed_content = df["content"].progress_apply(preprocess)
df["processed_content"] = processed_content
cv = CountVectorizer(max_features=2**11,min_df=10,stop_words='english') # count vectorizer
dtm = cv.fit_transform(df['processed_content'])
LDA = LatentDirichletAllocation(n_components=7,random_state=42,n_jobs=-1) # LDA topic Modeling Technique
LDA.fit(dtm)
# get the vocabulary length of words 
len(cv.get_feature_names())
# get some random words 
random_word_id = random.randint(0,2048)
cv.get_feature_names()[10]
# get the topics 
LDA.components_
LDA.components_.shape
single_topic = LDA.components_[0]
# get the last 10 values which has high prob
top_ten_words = single_topic.argsort()[-10:]
# from here we can see, that words like "Pay, company, coronavirus, worker " are related to business 
for index in top_ten_words:
    print(cv.get_feature_names()[index])
# get the highest probability words per topic 
for i,topic in enumerate(LDA.components_):
    print(f'The top 25 words for topic #{i}')
    print([cv.get_feature_names()[index] for index in topic.argsort()[-25:]])
    print('\n\n')

# Here we can check topic#1 related to financial, topic # 3 relates to tech, 
topic_results = LDA.transform(dtm)
df['Topic'] = topic_results.argmax(axis=1)
tfidf = TfidfVectorizer(max_features=2**11,min_df=10, stop_words='english')
dtm = tfidf.fit_transform(df['processed_content'])
nmf_model  = NMF(n_components=7,random_state=42)
nmf_model.fit(dtm)
tfidf.get_feature_names()[1480]
for i,topic in enumerate(nmf_model.components_):
    print(f'The top 25 words for topic #{i}')
    print([tfidf.get_feature_names()[index] for index in topic.argsort()[-25:]])
#  Here we can see that 
# nmfNumeric_to_topic = {0:'',1:'finance',2:'',3:'business',4:'healthcare',5:'',6:''}
#it depends on how we infer topics
topic_results = nmf_model.transform(dtm)
df['NMF_Topic'] = topic_results.argmax(axis=-1)
df.head()