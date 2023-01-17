# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



print(train_df.shape)

print(test_df.shape)
train_df.head()
test_df.head()
#print(train_df[train_df['question1']==""])

train_df[train_df['question2'].isnull()]
train_df[train_df['question1'].isnull()]
train_df.dropna(inplace=True)

print(len(train_df))
from nltk.corpus import stopwords

import string

from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm

from collections import OrderedDict as OD

import nltk



stop_words= list(stopwords.words('english'))

punc_list= list(string.punctuation)

lemma = WordNetLemmatizer()



def clean_docs(question):

    stop_words_cleaned= ' '.join([i for i in question.lower().split() if i not in stop_words])

    punc_cleaned=''.join([i for i in stop_words_cleaned if i not in punc_list])

    normalized= ' '.join([lemma.lemmatize(i) for i in punc_cleaned.split()])

    #unique_words= ' '.join(OD.fromkeys(normalized.split()))

    return normalized

    

"""

def clean_sentence(sentence):

    "POS tag the sentence and then lemmatize the words."

    global lemmatizer

    words = sentence.lower().split()

    pos_tagged = nltk.pos_tag(words)

    pos_tagged_stems = [(lemmatizer.lemmatize(i[0]), i[1]) for i in pos_tagged]

    return pos_tagged_stems

"""

"""

for i in tqdm(range(len(train_df['question1'])),mininterval=15):

    train_df.loc[train_df['question1']!=0,'q1_cleaned']=clean_docs(train_df['question1'][i])

"""

tqdm.pandas(mininterval=15, ncols=80, desc='Question1')

train_df['q1_clean'] = train_df.question1.progress_apply(clean_docs)



tqdm.pandas(mininterval=15, ncols=80, desc='Question2')

train_df['q2_clean'] = train_df.question2.progress_apply(clean_docs)

    

    
train_df.head()
train_df[train_df['is_duplicate']==1].head()
train_df[train_df['is_duplicate']==0].head()
train_df.isnull().any()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



tfidf= TfidfVectorizer(min_df=1)



similarity_list=[]



train_df['combined_q'] = list(zip(train_df['q1_clean'],train_df['q2_clean']))



def tf_idf(document):

    try:

        tfidf_matrix=tfidf.fit_transform(document)

        return tfidf_matrix

    except ValueError():

        return 0

tqdm.pandas(mininterval=15, ncols=80, desc='combined_q')

train_df['tf_idf']= train_df['combined_q'].progress_apply(tf_idf)





def get_cosine_sim(vectors):

    return float(cosine_similarity(vectors[0],vectors[1]).A)



train_df['tf_idf'].head()

    
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=1)



train_df['combined_q']= list(zip(train_df['q1_clean'],train_df['q2_clean']))



#print(train_df['combined_q'].head())



def tfidf(docs):

    try:

        tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

        return tfidf_matrix

    except ValueError:

        return 0

tqdm.pandas(mininterval=15, ncols=80, desc='combined_q')

train_df['tf_idf'] = train_df.combined_q.progress_apply(tfidf)



   
from sklearn.metrics.pairwise import cosine_similarity

#df_train['tf_idf'].drop()

def cosine_sim(vectors):

    try:

        return float(cosine_similarity(vectors[0],vectors[1]))

    except ValueError:

        return 0

tqdm.pandas(mininterval=15, ncols=80, desc='tf_idf')

train_df['cosine_sim'] = train_df.tf_idf.progress_apply(cosine_sim)

#print(cosine_sim(train_df['tf_idf'][1]))   

#print()

#print(cosine_similarity(train_df['tf_idf'][1][0],train_df['tf_idf'][1][1]))