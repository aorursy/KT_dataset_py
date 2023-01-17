



import pandas as pd

import numpy as np

import os

import re

import gensim

import spacy

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("german")







data_test = pd.read_csv('../input/10k-german-news-articles/Articles.csv')

data_test.head()


body_new = [re.sub('<[^<]+?>|&amp', '', word) for word in data_test['Body'].values]

data_test['Body'] = body_new

#! python -m spacy download de_core_news_sm

! python -m spacy download de_core_news_sm

import spacy

import de_core_news_sm

nlp = de_core_news_sm.load()





#nlp = spacy.load('de_core_news_sm')



## Drop Duplicates and NA values



data_test_clean = data_test.dropna()

data_test_clean = data_test_clean.drop_duplicates()
article_data = data_test_clean[['ID_Article','Title','Body']].drop_duplicates()

article_data.shape


## Convert the body text into a series of words to be fed into the model



def text_clean_tokenize(article_data):

    

    review_lines = list()



    lines = article_data['Body'].values.astype(str).tolist()



    for line in lines:

        tokens = word_tokenize(line)

        tokens = [w.lower() for w in tokens]

        table = str.maketrans('','',string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words('german'))

        words = [w for w in words if not w in stop_words]

        words = [stemmer.stem(w) for w in words]



        review_lines.append(words)

    return(review_lines)

    

    

review_lines = text_clean_tokenize(article_data)
## Building the word2vec model



model =  gensim.models.Word2Vec(sentences = review_lines,

                               size=100,

                               window=2,

                               workers=4,

                               min_count=2,

                               seed=42,

                               iter= 50)



model.save("word2vec.model")







word_list = list(model.wv.vocab)



for words in word_list[1:10]:

    print('Similar Words for :',words)

    

    print(model.wv.similar_by_word(words))

    print('--------------------------\n')











#print(word_list)



# Convert each article lines to word2vec representation

import spacy



def tokenize(sent):

    doc = nlp.tokenizer(sent)

    return [token.lower_ for token in doc if not token.is_punct]



new_df = (article_data['Body'].apply(tokenize).apply(pd.Series))



new_df = new_df.stack()

new_df = (new_df.reset_index(level=0)

                .set_index('level_0')

                .rename(columns={0: 'word'}))



new_df = new_df.join(article_data.drop('Body', 1), how='left')



new_df = new_df[['word','ID_Article']]

vectors = model.wv[word_list]

vectors_df = pd.DataFrame(vectors)

vectors_df['word'] = word_list

merged_frame = pd.merge(vectors_df, new_df, on='word')

merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby('ID_Article').mean().reset_index()

del merged_frame

del new_df

del vectors

merged_frame_rolled_up.head()
from sklearn.metrics.pairwise import cosine_similarity

cosine_matrix = pd.DataFrame(cosine_similarity(merged_frame_rolled_up))

cosine_matrix.columns = list(merged_frame_rolled_up['ID_Article'])

cosine_matrix.head()


reco_articles = {}

i = 0

for col_name in cosine_matrix.columns:

    tmp = cosine_matrix[[col_name]].sort_values(by=col_name,ascending=False)

    tmp = tmp.iloc[1:]

    tmp = tmp.head(20)

    recommended_articles = list(article_data[article_data['ID_Article'].isin(tmp.index)]['Title'].values)

    chosen_article = list(article_data[article_data['ID_Article']==col_name]['Title'].values)

    tmp = {'Chosen-Articles': len(recommended_articles)* chosen_article,'Recommended-Articles':recommended_articles}

    reco_articles[i] = tmp

    i = i+1

    del tmp

print('Ended')

    

    

    

    

    
## Convert Dictionary Object to a data frame



df_reco = pd.concat([pd.DataFrame(v) for k, v in reco_articles.items()])

df_reco.head()



## Making sure that the same articles do not get recommended



df_reco = df_reco[df_reco['Chosen-Articles']!=df_reco['Recommended-Articles']]
import random



list_of_articles = df_reco['Chosen-Articles'].values

random.shuffle(list_of_articles)

list_of_articles = list_of_articles[:9]



for article in list_of_articles:

    tmp = df_reco[df_reco['Chosen-Articles']==article]

    print('--------------------------------------- \n')

    print('Recommendation for ',article,' is :')

    print('Recommended Articles')

    print(tmp['Recommended-Articles'].values)

    
