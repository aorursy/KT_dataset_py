import os

import datetime as dt

import math

import numpy as np

import pandas as pd
# Input data files are available in the "../input/" directory.

input_dir = "../input"



questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])



answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])



comments = pd.read_csv(os.path.join(input_dir, 'comments.csv'))

comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'])
questions['questions_date_added'].min()
questions.head(3)
answers.head(3)
comments.head(3)
questions['questions_text'] = questions['questions_title'] + ' ' + questions['questions_body']

questions_text = questions[['questions_id', 'questions_text']]

questions_text.head(5)
questions_text.to_parquet('questions_text.parquet.gzip', compression='gzip')
test_period_start = dt.datetime(2018, 7, 1)

test_period_end = dt.datetime(2019, 1, 31)
train_questions = questions[questions['questions_date_added'] < test_period_start]

train_questions['text'] = train_questions['questions_title'] + ' ' + train_questions['questions_body']

train_questions_by_users = train_questions.groupby('questions_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_questions_by_users = train_questions_by_users.rename(columns={'questions_author_id': 'user_id'})
train_answers= answers[answers['answers_date_added'] < test_period_start]

train_answers['text'] = train_answers['answers_body'].apply(str)

train_answers_by_users = train_answers.groupby('answers_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_answers_by_users = train_answers_by_users.rename(columns={'answers_author_id': 'user_id'})
train_comments = comments[comments['comments_date_added'] < test_period_start]

train_comments['text'] = train_comments['comments_body'].apply(str)

train_comments_by_users = train_comments.groupby('comments_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_comments_by_users = train_comments_by_users.rename(columns={'comments_author_id': 'user_id'})
direct_user_texts = pd.concat([train_questions_by_users, train_answers_by_users, train_comments_by_users], axis=0)

direct_user_texts = direct_user_texts.groupby('user_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

print(direct_user_texts.shape)

direct_user_texts.head(3)

direct_user_texts.to_parquet('direct_user_texts.parquet.gzip', compression='gzip')
train_questions_to_answers_by_users = train_questions.merge(

    train_answers[['answers_question_id', 'answers_author_id']], 

    left_on='questions_id', right_on='answers_question_id', how='inner').groupby(

    'answers_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_questions_to_answers_by_users = train_questions_to_answers_by_users.rename(columns={'answers_author_id': 'user_id'})

train_questions_to_answers_by_users.shape
train_answers_to_questions_by_users = train_answers.merge(

    train_questions[['questions_id', 'questions_author_id']], 

    left_on='answers_question_id', right_on='questions_id', how='inner').groupby(

    'questions_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_answers_to_questions_by_users = train_answers_to_questions_by_users.rename(columns={'questions_author_id': 'user_id'})

train_answers_to_questions_by_users.shape
train_questions_to_comments_by_users = train_questions.merge(

    train_comments[['comments_parent_content_id', 'comments_author_id']], 

    left_on='questions_id', right_on='comments_parent_content_id', how='inner').groupby(

    'comments_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_questions_to_comments_by_users = train_questions_to_comments_by_users.rename(columns={'comments_author_id': 'user_id'})

train_questions_to_comments_by_users.shape
train_comments_to_questions_by_users = train_comments.merge(

    train_questions[['questions_id', 'questions_author_id']], 

    left_on='comments_parent_content_id', right_on='questions_id', how='inner').groupby(

    'questions_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_comments_to_questions_by_users = train_comments_to_questions_by_users.rename(columns={'questions_author_id': 'user_id'})

train_comments_to_questions_by_users.shape
train_answers_to_comments_by_users = train_answers.merge(

    train_comments[['comments_parent_content_id', 'comments_author_id']], 

    left_on='answers_id', right_on='comments_parent_content_id', how='inner').groupby(

    'comments_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_answers_to_comments_by_users = train_answers_to_comments_by_users.rename(columns={'comments_author_id': 'user_id'})

train_answers_to_comments_by_users.shape
train_comments_to_answers_by_users = train_comments.merge(

    train_answers[['answers_id', 'answers_author_id']], 

    left_on='comments_parent_content_id', right_on='answers_id', how='inner').groupby(

    'answers_author_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

train_comments_to_answers_by_users = train_comments_to_answers_by_users.rename(columns={'answers_author_id': 'user_id'})

train_comments_to_answers_by_users.shape
indirect_user_texts = pd.concat(

    [train_questions_to_answers_by_users, train_answers_to_questions_by_users,

     train_questions_to_comments_by_users, train_comments_to_questions_by_users,

     train_answers_to_comments_by_users, train_comments_to_answers_by_users], axis=0)

indirect_user_texts = indirect_user_texts.groupby('user_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

print(indirect_user_texts.shape)

indirect_user_texts.head(3)

indirect_user_texts.to_parquet('indirect_user_texts.parquet.gzip', compression='gzip')
merged_user_texts = pd.concat([direct_user_texts, indirect_user_texts], axis=0)

merged_user_texts = merged_user_texts.groupby('user_id')['text'].apply(lambda texts: '\n '.join(texts)).reset_index()

print(merged_user_texts.shape)

merged_user_texts.head(3)

merged_user_texts.to_parquet('merged_user_texts.parquet.gzip', compression='gzip')
direct_user_texts['text'][0]
import spacy

nlp = spacy.load('en')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')



from wordcloud import WordCloud



import gensim
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']



def nlp_preprocessing(data):

    """ Use NLP to transform the text corpus to cleaned sentences and word tokens

    """    

    

    def token_filter(token):

        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list

        """    

        return not token.is_stop and token.is_alpha and token.pos_ in token_pos

    

    processed_tokens = []

    data_pipe = nlp.pipe(data)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        processed_tokens.append(filtered_tokens)

    return processed_tokens    
# Tokenize text

lsi_tokens = nlp_preprocessing(direct_user_texts['text'])



# Create vocabulary

remove_terms_below_document_numbers = 10

remove_terms_above_corpus_size = 0.5

vocabulary_size = 10000

lsi_dic = gensim.corpora.Dictionary(lsi_tokens)

lsi_dic.filter_extremes(no_below=remove_terms_below_document_numbers, 

                        no_above=remove_terms_above_corpus_size, 

                        keep_n=vocabulary_size)

lsi_dic.save('lsi_dic_size_{}.model'.format(vocabulary_size))
print('Dictionary size: {}'.format(len(lsi_dic)))
# Convert tokens to gensim document format

lsi_corpus = [lsi_dic.doc2bow(doc) for doc in lsi_tokens]

# Apply tfidf on the corpus first to remove the effect of function words (those with high document frequencies but low tf-idf scores)

lsi_tfidf = gensim.models.TfidfModel(lsi_corpus)

lsi_tfidf.save('lsi_corpus_voc_size_{}.model'.format(vocabulary_size))

lsi_corpus = lsi_tfidf[lsi_corpus]
num_topics = 50

lsi_model = gensim.models.LsiModel(lsi_corpus, id2word=lsi_dic, num_topics=num_topics)

lsi_model.save('lsi_topics_{}.model'.format(num_topics))
def compute_lsi_probs(doc, text_col, num_topics):

    doc_tokens = nlp_preprocessing([doc[text_col]])[0]

    doc_bow = lsi_tfidf[lsi_dic.doc2bow(doc_tokens)]

    doc_lsi = lsi_model[doc_bow]

    doc_scores = np.zeros(shape=num_topics)

    for v,k in doc_lsi:

        doc_scores[v] = k

    return doc_scores



def lsi_similarity(lsi_1, lsi_2):

    return np.dot(lsi_1,lsi_2) / np.sqrt(np.dot(lsi_1,lsi_1) * np.dot(lsi_2,lsi_2))
text_col = 'text'

doc1_scores = compute_lsi_probs(direct_user_texts.iloc[0], text_col, num_topics)

doc2_scores = compute_lsi_probs(direct_user_texts.iloc[0], text_col, num_topics)

lsi_similarity(doc1_scores, doc2_scores)
doc1_scores = compute_lsi_probs(direct_user_texts.iloc[0], text_col, num_topics)

doc2_scores = compute_lsi_probs(direct_user_texts.iloc[1], text_col, num_topics)

lsi_similarity(doc1_scores, doc2_scores)
doc1_scores = compute_lsi_probs(direct_user_texts.iloc[0], text_col, num_topics)

doc2_scores = compute_lsi_probs(direct_user_texts.iloc[2], text_col, num_topics)

lsi_similarity(doc1_scores, doc2_scores)
def compute_lsi_scores(doc, text_col, num_topics):

    doc_tokens = nlp_preprocessing([doc[text_col]])[0]

    doc_bow = lsi_tfidf[lsi_dic.doc2bow(doc_tokens)]

    doc_lsi = lsi_model[doc_bow]

    doc_scores = np.zeros(shape=num_topics)

    for v,k in doc_lsi:

        doc_scores[v] = k

    return pd.Series(doc_scores)
print(questions_text.shape)

questions_topics = questions_text.apply(compute_lsi_scores, axis=1, text_col='questions_text', num_topics=num_topics)

print(questions_topics.shape)
questions_topics = questions_topics.set_index(questions_text['questions_id'])

questions_topics.tail(2)
questions_topics.columns = ['Topic_{}'.format(i) for i in questions_topics.columns]

questions_topics.to_parquet('questions_topics_vs_{}_nt_{}.parquet.gzip'.format(vocabulary_size, num_topics), compression='gzip')
print(merged_user_texts.shape)

merged_user_topics = merged_user_texts.apply(compute_lsi_scores, axis=1, text_col='text', num_topics=num_topics)

print(merged_user_topics.shape)
merged_user_topics = merged_user_topics.set_index(merged_user_texts['user_id'])

merged_user_topics.tail(2)
merged_user_topics.columns = ['Topic_{}'.format(i) for i in merged_user_topics.columns]

merged_user_topics.to_parquet('merged_user_topics_vs_{}_nt_{}.parquet.gzip'.format(vocabulary_size, num_topics), compression='gzip')
os.listdir()