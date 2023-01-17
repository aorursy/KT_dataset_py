import numpy as np # linear algebra

import pandas as pd #

import numpy as np

import os

import re

import gensim

import spacy

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



biorxiv = pd.read_csv("/kaggle/input/clean-csv/biorxiv_clean.csv")

biorxiv.shape

biorxiv.head()



biorxiv = biorxiv[['paper_id','title','text']].dropna().drop_duplicates()

pmc = pd.read_csv('/kaggle/input/clean-csv-new/clean_pmc.csv')

pmc = pmc[['paper_id','title','text']].dropna().drop_duplicates()



biorxiv = pd.concat([biorxiv,pmc]).drop_duplicates()



biorxiv = biorxiv.sample(n=300)



biorxiv.head()

biorxiv_split = pd.concat([pd.Series(row['paper_id'], row['text'].split('.')) for _, row in biorxiv.iterrows()]).reset_index()
biorxiv_split.columns = ['sentences','paper_id']

biorxiv_split = biorxiv_split.replace('\n','', regex=True)
! python -m spacy download en_core_web_sm

import spacy

import en_core_web_sm

nlp = en_core_web_sm.load()


stemmer = SnowballStemmer("english")



def text_clean_tokenize(article_data):

    

    review_lines = list()



    lines = article_data['text'].values.astype(str).tolist()



    for line in lines:

        tokens = word_tokenize(line)

        tokens = [w.lower() for w in tokens]

        table = str.maketrans('','',string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words('english'))

        words = [w for w in words if not w in stop_words]

        words = [stemmer.stem(w) for w in words]



        review_lines.append(words)

    return(review_lines)

    

    

review_lines = text_clean_tokenize(biorxiv)
model =  gensim.models.Word2Vec(sentences = review_lines,

                               size=1000,

                               window=2,

                               workers=4,

                               min_count=2,

                               seed=42,

                               iter= 50)



model.save("word2vec.model")
import spacy

nlp = en_core_web_sm.load()

def tokenize(sent):

    doc = nlp.tokenizer(sent)

    return [token.lower_ for token in doc if not token.is_punct]



new_df = (biorxiv_split['sentences'].apply(tokenize).apply(pd.Series))



new_df = new_df.stack()

new_df = (new_df.reset_index(level=0)

                .set_index('level_0')

                .rename(columns={0: 'word'}))



new_df = new_df.join(biorxiv_split,how='left')



new_df = new_df[['word','paper_id','sentences']]

word_list = list(model.wv.vocab)

vectors = model.wv[word_list]

vectors_df = pd.DataFrame(vectors)

vectors_df['word'] = word_list

merged_frame = pd.merge(vectors_df, new_df, on='word')

merged_frame_rolled_up = merged_frame.drop('word',axis=1).groupby(['paper_id','sentences']).mean().reset_index()

del merged_frame

del new_df

del vectors
questions = {

    'questions' : ["What is known about transmission, incubation, and environmental stability of COVID?",

                "What do we know about COVID risk factors?","What do we know about virus genetics, origin, and evolution of COVID?","What do we know about vaccines and therapeutics for COVID?"]

}

questions = pd.DataFrame(questions)
new_df = (questions['questions'].apply(tokenize).apply(pd.Series))



new_df = new_df.stack()

new_df = (new_df.reset_index(level=0)

                .set_index('level_0')

                .rename(columns={0: 'word'}))



new_df = new_df.join(questions,how='left')



new_df = new_df[['word','questions']]

word_list = list(model.wv.vocab)

vectors = model.wv[word_list]

vectors_df = pd.DataFrame(vectors)

vectors_df['word'] = word_list

merged_frame = pd.merge(vectors_df, new_df, on='word')

question2vec = merged_frame.drop('word',axis=1).groupby(['questions']).mean().reset_index()
from numpy import dot

from numpy.linalg import norm





for i in range(len(question2vec)):

    tmp = question2vec.iloc[[i]]

    tmp = tmp.drop('questions',axis=1)

    a = np.array(tmp.values)

    list_of_scores = []

    for j in range(len(merged_frame_rolled_up)):

        tmp_ = merged_frame_rolled_up.iloc[[j]]

        tmp_ = tmp_.drop(['paper_id','sentences'],axis=1)

        b = np.array(tmp_.values)

        b = b.T

        cos_sim = dot(a, b)/(norm(a)*norm(b))

        list_of_scores.append(float(cos_sim))

    df_answer = pd.DataFrame()

    df_answer['sentence'] = merged_frame_rolled_up['sentences'].tolist()

    df_answer['scores'] = list_of_scores

    df_answer['question'] = question2vec.iloc[i]['questions']

    df_answer.sort_values(by='scores',ascending=False,inplace=True)

    print('---------------------------- \n')

    print('\n Answers for question: \n')

    print(question2vec.iloc[i]['questions'])

    print(df_answer.head(10)['sentence'].values)

        

        