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
import re

import nltk

import gensim

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from gensim.models import CoherenceModel

from  more_itertools import unique_everseen

from operator import itemgetter

from gensim.models.nmf import Nmf

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF
data = pd.read_csv("/kaggle/input/unstructured-l0-nlp-hackathon/data.csv")

data.head()
# cleaning

# remove punctuation and special characters

data['clean_text'] = data['text'].apply(lambda x: re.sub(r'[^A-Za-z\. ]',' ',x))



# convert to lowercase

data['clean_text'] = data['clean_text'].str.lower()





data['clean_text'] = data['clean_text'].apply(lambda x: re.sub(r'[\.]+','.',x))

data['clean_text'] = data['clean_text'].apply(lambda x: re.sub(r'[ ]+',' ',x))



data.head()
def getWordNetTags(originalTag):

    """ Converts tags to WordNet tags. """

    

    tag = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    try:

        return tag[originalTag[0]]

    except:

        return wordnet.NOUN
# removing stop words and typos and lemmatizing

stopwordsList = stopwords.words('english')

words = set(nltk.corpus.words.words())

lemmatizer = WordNetLemmatizer()



data['clean_textList'] = data['clean_text'].apply(lambda x: [word for word in x.split() if word not in stopwordsList and word in words])

data['pos_tag'] = data['clean_textList'].apply(lambda x: nltk.pos_tag(x))

data['clean_textList'] = data['pos_tag'].apply(lambda x: [lemmatizer.lemmatize(word[0],getWordNetTags(word[1])) for word in x ])

#data['clean_textList'] = data['pos_tag'].apply(lambda x: [lemmatizer.lemmatize(word[0],getWordNetTags(word[1])) for word in x if word[1][0]=='N'])

data['clean_textList'] = data['clean_textList'].apply(lambda x: [word for word in x if len(word)>2])

data.head()
allWords = data['clean_textList'].apply(pd.Series).stack().tolist()

len(allWords),len(set(allWords))
#data['clean_textListSubset'] = data['clean_textList'].apply(lambda x: [word for word in x if word not in most_frequent_list])

#data['clean_textListSubset'] = data['clean_textList'].apply(lambda x: [word for word in x if word not in least_frequent_list])

data['clean_text'] = data['clean_textList'].apply(lambda x: ' '.join(x))

data.head()
# NMF



texts = data['clean_textList']



dictionary = gensim.corpora.Dictionary(texts)



# filtering out frequency extremes to limit the number of features

dictionary.filter_extremes(

    no_below=5,

    no_above=0.85,

    keep_n=5000

)



# bag-of-words

corpus = [dictionary.doc2bow(text) for text in texts]



# To get the best num of topics, calculating coherence score

topic_nums = [5,6,7,8,9]



coherence_scores = []



for num in topic_nums:

    nmf = Nmf(

        corpus=corpus,

        num_topics=num,

        id2word=dictionary,

        chunksize=2000,

        passes=5,

        kappa=.1,

        minimum_probability=0.01,

        w_max_iter=300,

        w_stop_condition=0.0001,

        h_max_iter=100,

        h_stop_condition=0.001,

        eval_every=10,

        normalize=True,

        random_state=42

    )



    cm = CoherenceModel(

        model=nmf,

        texts=texts,

        dictionary=dictionary,

        coherence='c_v'

    )

    print(num,'topics')

    

    coherence_scores.append(round(cm.get_coherence(), 5))



scores = list(zip(topic_nums, coherence_scores))

best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
scores,best_num_topics
# calculate for nest num of topics

tfidf_vectorizer = TfidfVectorizer(

    min_df=5,

    max_df=0.85,

    max_features=5000,

    ngram_range=(1, 2),

    preprocessor=' '.join

)



tfidf = tfidf_vectorizer.fit_transform(texts)



# Save the feature names for later to create topic summaries

tfidf_fn = tfidf_vectorizer.get_feature_names()



# Run the nmf model

nmf = NMF(

    n_components=best_num_topics,

    init='nndsvd',

    max_iter=500,

    l1_ratio=0.0,

    solver='cd',

    alpha=0.0,

    tol=1e-4,

    random_state=42

).fit(tfidf)
def topic_table(model, feat_names, n_top_words):

    

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.

    

    word_dict = {};

    for i in range(best_num_topics):

        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1] 

        #get the top 20 word ids for each topic 

        words = [feat_names[key] for key in words_ids] #obtain the word using the id

        word_dict['Topic ' + '{:01d}'.format(i+1)] = words

    return pd.DataFrame(word_dict)
# creating summary

docweights = nmf.transform(tfidf_vectorizer.transform(texts))



n_top_words = 9



topic_df = topic_table(

    nmf,

    tfidf_fn,

    n_top_words

)

topic_df= topic_df.T

#Summary

topic_df['topics'] = topic_df.apply(lambda x: ' '.join(x), axis=1)

topic_df['topics'] = topic_df['topics'].apply(lambda x: list(unique_everseen(x.split(' '))))

topic_df['topics'] = topic_df['topics'].apply(lambda x: ' '.join(x))

topic_df = topic_df['topics'].reset_index().reset_index()

topic_df.drop(columns=['index'],inplace=True)

topic_df.columns = ['topic_num', 'topics']



topic_df['topic'] = ["room_rentals",'glassdoor_reviews','room_rentals','Automobiles','glassdoor_reviews','sports_news']

topic_df
# scoring



Id = data['Id'].tolist()



df_temp = pd.DataFrame({

    'Id': Id,

    'topic_num': docweights.argmax(axis=1)

})



# Merging to get the topic num with url

merged_topic = df_temp.merge(

    topic_df,

    on='topic_num',

    how='left'

)



# Merging with the original df

df_topics = pd.merge(

    data,

    merged_topic,

    on='Id',

    how='left'

)





df_topics.head()
df_topics[["Id","topic"]].to_csv("sample_submission.csv",index=False)