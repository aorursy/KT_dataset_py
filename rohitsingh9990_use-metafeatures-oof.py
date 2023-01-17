import os

import re

import gc

import pickle  

import random

import keras



import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K



from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda

from keras.optimizers import Adam

from keras.callbacks import Callback

from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold

from sklearn.linear_model import MultiTaskElasticNet

from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))

import string



seed(42)

tf.random.set_seed(42)

random.seed(42)
data_dir = '../input/google-quest-challenge/'

train = pd.read_csv(path_join(data_dir, 'train.csv'))

test = pd.read_csv(path_join(data_dir, 'test.csv'))

print(train.shape, test.shape)

train.head()
targets = [

        'question_asker_intent_understanding',

        'question_body_critical',

        'question_conversational',

        'question_expect_short_answer',

        'question_fact_seeking',

        'question_has_commonly_accepted_answer',

        'question_interestingness_others',

        'question_interestingness_self',

        'question_multi_intent',

        'question_not_really_a_question',

        'question_opinion_seeking',

        'question_type_choice',

        'question_type_compare',

        'question_type_consequence',

        'question_type_definition',

        'question_type_entity',

        'question_type_instructions',

        'question_type_procedure',

        'question_type_reason_explanation',

        'question_type_spelling',

        'question_well_written',

        'answer_helpful',

        'answer_level_of_information',

        'answer_plausible',

        'answer_relevance',

        'answer_satisfaction',

        'answer_type_instructions',

        'answer_type_procedure',

        'answer_type_reason_explanation',

        'answer_well_written'    

    ]



input_columns = ['question_title', 'question_body', 'answer']
find = re.compile(r"^[^.]*")



train['netloc'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

test['netloc'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



# train['self_answered'] =  train.apply(lambda x: 1 if x.question_user_name == x.answer_user_name else 0, axis=1)

# test['self_answered'] =  test.apply(lambda x: 1 if x.question_user_name == x.answer_user_name else 0, axis=1)





features = ['netloc', 'category']

merged = pd.concat([train[features], test[features]])

ohe = OneHotEncoder()

ohe.fit(merged)



features_train = ohe.transform(train[features]).toarray()

features_test = ohe.transform(test[features]).toarray()
module_url = "../input/universalsentenceencoderlarge4/"

embed = hub.load(module_url)
# https://stackoverflow.com/a/47091490/4084039

import re

from tqdm import tqdm

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize



ps = PorterStemmer()



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase





def preprocess_text(df, column_name):

    preprocessed = []

    # tqdm is for printing the status bar

    for sentance in tqdm(df[column_name].values):

        sent = decontracted(sentance)

        sent = sent.replace('\\r', ' ')

        sent = sent.replace('\\"', ' ')

        sent = sent.replace('\\n', ' ')

        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

        sent = ' '.join(e for e in sent.split() if e.lower() not in stop_words)

        # porter stemming to root word

#         sent = ' '.join([ps.stem(word) for word in sent.split()])                    

        preprocessed.append(sent.lower().strip())

    return preprocessed
%%time



embeddings_train = {}

embeddings_test = {}

for text in input_columns:

    print(text)

    

#     train_text = preprocess_text(train, text)

#     test_text = preprocess_text(test, text)

    

    train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()

    test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()

    

    curr_train_emb = []

    curr_test_emb = []

    batch_size = 4

    ind = 0

    while ind*batch_size < len(train_text):

        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())

        ind += 1

        

    ind = 0

    while ind*batch_size < len(test_text):

        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())

        ind += 1    

        

    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)

    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)

    

# del embed

K.clear_session()

gc.collect()
# num of words in text

train['num_words_ques_title'] = train['question_title'].apply(lambda x: len(x.split()))

train['num_words_ques_body'] = train['question_body'].apply(lambda x: len(x.split()))

train['num_words_answer'] = train['answer'].apply(lambda x: len(x.split()))



test['num_words_ques_title'] = test['question_title'].apply(lambda x: len(x.split()))

test['num_words_ques_body'] = test['question_body'].apply(lambda x: len(x.split()))

test['num_words_answer'] = test['answer'].apply(lambda x: len(x.split()))





# num of unique words in text

train['num_uniq_words_ques_title'] = train['question_title'].apply(lambda x: len(np.unique(x.split())))

train['num_uniq_words_ques_body'] = train['question_body'].apply(lambda x: len(np.unique(x.split())))

train['num_uniq_words_answer'] = train['answer'].apply(lambda x: len(np.unique(x.split())))



test['num_uniq_words_ques_title'] = test['question_title'].apply(lambda x: len(np.unique(x.split())))

test['num_uniq_words_ques_body'] = test['question_body'].apply(lambda x: len(np.unique(x.split())))

test['num_uniq_words_answer'] = test['answer'].apply(lambda x: len(np.unique(x.split())))





# # num of characters in text

# train['num_chars_ques_title'] = train['question_title'].apply(lambda x: len(x))

# train['num_chars_ques_body'] = train['question_body'].apply(lambda x: len(x))

# train['num_chars_answer'] = train['answer'].apply(lambda x: len(x))



# test['num_chars_ques_title'] = test['question_title'].apply(lambda x: len(x))

# test['num_chars_ques_body'] = test['question_body'].apply(lambda x: len(x))

# test['num_chars_answer'] = test['answer'].apply(lambda x: len(x))





# num of stop_words in text

train['num_stop_words_ques_title'] = train['question_title'].apply(lambda x: len([word for word in x.split() if word in stop_words]))

train['num_stop_words_ques_body'] = train['question_body'].apply(lambda x: len([word for word in x.split() if word in stop_words]))

train['num_stop_words_answer'] = train['answer'].apply(lambda x: len([word for word in x.split() if word in stop_words]))





test['num_stop_words_ques_title'] = test['question_title'].apply(lambda x: len([word for word in x.split() if word in stop_words]))

test['num_stop_words_ques_body'] = test['question_body'].apply(lambda x: len([word for word in x.split() if word in stop_words]))

test['num_stop_words_answer'] = test['answer'].apply(lambda x: len([word for word in x.split() if word in stop_words]))





# num of punctuations in text

train['num_puncts_ques_title'] = train['question_title'].apply(lambda x: len([char for char in x if char in string.punctuation]))

train['num_puncts_ques_body'] = train['question_body'].apply(lambda x: len([char for char in x if char in string.punctuation]))

train['num_puncts_answer'] = train['answer'].apply(lambda x: len([char for char in x if char in string.punctuation]))



test['num_puncts_ques_title'] = test['question_title'].apply(lambda x: len([char for char in x if char in string.punctuation]))

test['num_puncts_ques_body'] = test['question_body'].apply(lambda x: len([char for char in x if char in string.punctuation]))

test['num_puncts_answer'] = test['answer'].apply(lambda x: len([char for char in x if char in string.punctuation]))





# # num of upper case words in text

# train['num_upper_words'] = train['text'].apply(lambda x: len([word for word in x.split() if word.isupper()]))

# test['num_upper_words'] = test['text'].apply(lambda x: len([word for word in x.split() if word.isupper()]))





# # num of title case words in text

# train['num_title_words'] = train['text'].apply(lambda x: len([word for word in x.split() if word.istitle()]))

# test['num_title_words'] = test['text'].apply(lambda x: len([word for word in x.split() if word.istitle()]))





# Average length of words in text

train['mean_len_words_ques_title'] = train['question_title'].apply(lambda x: np.mean([len(word) for word in x.split()]))

train['mean_len_words_ques_body'] = train['question_body'].apply(lambda x: np.mean([len(word) for word in x.split()]))

train['mean_len_words_ques_answer'] = train['answer'].apply(lambda x: np.mean([len(word) for word in x.split()]))





test['mean_len_words_ques_title'] = test['question_title'].apply(lambda x: np.mean([len(word) for word in x.split()]))

test['mean_len_words_ques_body'] = test['question_body'].apply(lambda x: np.mean([len(word) for word in x.split()]))

test['mean_len_words_ques_answer'] = test['answer'].apply(lambda x: np.mean([len(word) for word in x.split()]))





# Adding text sentiment features



analyzer = SentimentIntensityAnalyzer()



# # pos sentiment

train['text_sent_pos_ques_title'] = train['question_title'].apply(lambda x: analyzer.polarity_scores(x)['pos'])

train['text_sent_pos_ques_body'] = train['question_body'].apply(lambda x: analyzer.polarity_scores(x)['pos'])

train['text_sent_pos_answer'] = train['answer'].apply(lambda x: analyzer.polarity_scores(x)['pos'])



test['text_sent_pos_ques_title'] = test['question_title'].apply(lambda x: analyzer.polarity_scores(x)['pos'])

test['text_sent_pos_ques_body'] = test['question_body'].apply(lambda x: analyzer.polarity_scores(x)['pos'])

test['text_sent_pos_answer'] = test['answer'].apply(lambda x: analyzer.polarity_scores(x)['pos'])



# neg sentiment

train['text_sent_neg_ques_title'] = train['question_title'].apply(lambda x: analyzer.polarity_scores(x)['neg'])

train['text_sent_neg_ques_body'] = train['question_body'].apply(lambda x: analyzer.polarity_scores(x)['neg'])

train['text_sent_neg_answer'] = train['answer'].apply(lambda x: analyzer.polarity_scores(x)['neg'])



test['text_sent_neg_ques_title'] = test['question_title'].apply(lambda x: analyzer.polarity_scores(x)['neg'])

test['text_sent_neg_ques_body'] = test['question_body'].apply(lambda x: analyzer.polarity_scores(x)['neg'])

test['text_sent_neg_answer'] = test['answer'].apply(lambda x: analyzer.polarity_scores(x)['neg'])



# neu sentiment

train['text_sent_neu_ques_title'] = train['question_title'].apply(lambda x: analyzer.polarity_scores(x)['neu'])

train['text_sent_neu_ques_body'] = train['question_body'].apply(lambda x: analyzer.polarity_scores(x)['neu'])

train['text_sent_neu_answer'] = train['answer'].apply(lambda x: analyzer.polarity_scores(x)['neu'])



test['text_sent_neu_ques_title'] = test['question_title'].apply(lambda x: analyzer.polarity_scores(x)['neu'])

test['text_sent_neu_ques_body'] = test['question_body'].apply(lambda x: analyzer.polarity_scores(x)['neu'])

test['text_sent_neu_answer'] = test['answer'].apply(lambda x: analyzer.polarity_scores(x)['neu'])

from sklearn.preprocessing import StandardScaler
def standardize_num_features(feat_name):

    standardizer = StandardScaler(with_mean=False)

    embeddings_train[feat_name] = standardizer.fit_transform(train[feat_name].values.reshape(-1,1))

    embeddings_test[feat_name] = standardizer.transform(test[feat_name].values.reshape(-1,1))





standardize_num_features('num_words_ques_title') 

standardize_num_features('num_words_ques_body') 

standardize_num_features('num_words_answer')





standardize_num_features('num_uniq_words_ques_title')

standardize_num_features('num_words_ques_body')

standardize_num_features('num_uniq_words_answer')





standardize_num_features('num_stop_words_ques_title')

standardize_num_features('num_stop_words_ques_body')

standardize_num_features('num_stop_words_answer')



standardize_num_features('num_puncts_ques_title')

standardize_num_features('num_puncts_ques_body')

standardize_num_features('num_puncts_answer')





standardize_num_features('text_sent_pos_ques_title')

standardize_num_features('text_sent_pos_ques_body')

standardize_num_features('text_sent_pos_answer')



standardize_num_features('text_sent_neg_ques_title')

standardize_num_features('text_sent_neg_ques_body')

standardize_num_features('text_sent_neg_answer')



standardize_num_features('text_sent_neu_ques_title')

standardize_num_features('text_sent_neu_ques_body')

standardize_num_features('text_sent_neu_answer')





# # num words in text

# # embeddings_train['num_words_ques_title'] = standardizer.fit_transform(train['num_words_ques_title'].values.reshape(-1,1))

# # embeddings_test['num_words_ques_title'] = standardizer.transform(test['num_words_ques_title'].values.reshape(-1,1))

# # embeddings_train['num_words_ques_body'] = standardizer.fit_transform(train['num_words_ques_body'].values.reshape(-1,1))

# # embeddings_test['num_words_ques_body'] = standardizer.transform(test['num_words_ques_body'].values.reshape(-1,1))

# embeddings_train['num_words_answer'] = standardizer.fit_transform(train['num_words_answer'].values.reshape(-1,1))

# embeddings_test['num_words_answer'] = standardizer.transform(test['num_words_answer'].values.reshape(-1,1))



# # num unique words in text

# # embeddings_train['num_uniq_words_ques_title'] = standardizer.fit_transform(train['num_uniq_words_ques_title'].values.reshape(-1,1))

# # embeddings_test['num_uniq_words_ques_title'] = standardizer.transform(test['num_uniq_words_ques_title'].values.reshape(-1,1))

# # embeddings_train['num_uniq_words_ques_body'] = standardizer.fit_transform(train['num_uniq_words_ques_body'].values.reshape(-1,1))

# # embeddings_test['num_uniq_words_ques_body'] = standardizer.transform(test['num_uniq_words_ques_body'].values.reshape(-1,1))

# embeddings_train['num_uniq_words_answer'] = standardizer.fit_transform(train['num_uniq_words_answer'].values.reshape(-1,1))

# embeddings_test['num_uniq_words_answer'] = standardizer.transform(test['num_uniq_words_answer'].values.reshape(-1,1))



# # num of stop_words in text

# # embeddings_train['num_stop_words_ques_title'] = standardizer.fit_transform(train['num_stop_words_ques_title'].values.reshape(-1,1))

# # embeddings_test['num_stop_words_ques_title'] = standardizer.transform(test['num_stop_words_ques_title'].values.reshape(-1,1))

# # embeddings_train['num_stop_words_ques_body'] = standardizer.fit_transform(train['num_stop_words_ques_body'].values.reshape(-1,1))

# # embeddings_test['num_stop_words_ques_body'] = standardizer.transform(test['num_stop_words_ques_body'].values.reshape(-1,1))

# embeddings_train['num_stop_words_answer'] = standardizer.fit_transform(train['num_stop_words_answer'].values.reshape(-1,1))

# embeddings_test['num_stop_words_answer'] = standardizer.transform(test['num_stop_words_answer'].values.reshape(-1,1))



# # num of puncts in text

# # embeddings_train['num_puncts_ques_title'] = standardizer.fit_transform(train['num_puncts_ques_title'].values.reshape(-1,1))

# # embeddings_test['num_puncts_ques_title'] = standardizer.transform(test['num_puncts_ques_title'].values.reshape(-1,1))

# # embeddings_train['num_puncts_ques_body'] = standardizer.fit_transform(train['num_puncts_ques_body'].values.reshape(-1,1))

# # embeddings_test['num_puncts_ques_body'] = standardizer.transform(test['num_puncts_ques_body'].values.reshape(-1,1))

# embeddings_train['num_puncts_answer'] = standardizer.fit_transform(train['num_puncts_answer'].values.reshape(-1,1))

# embeddings_test['num_puncts_answer'] = standardizer.transform(test['num_puncts_answer'].values.reshape(-1,1))



# embeddings_train['text_sent_pos_ques_title'] = standardizer.fit_transform(train['text_sent_pos_ques_title'].values.reshape(-1,1))

# embeddings_test['text_sent_pos_ques_title'] = standardizer.transform(test['text_sent_pos_ques_title'].values.reshape(-1,1))





# embeddings_train['text_sent_neg_ques_title'] = standardizer.fit_transform(train['text_sent_neg_ques_title'].values.reshape(-1,1))

# embeddings_test['text_sent_neg_ques_title'] = standardizer.transform(test['text_sent_neg_ques_title'].values.reshape(-1,1))



# embeddings_train['text_sent_neu_ques_title'] = standardizer.fit_transform(train['text_sent_neu_ques_title'].values.reshape(-1,1))

# embeddings_test['text_sent_neu_ques_title'] = standardizer.transform(test['text_sent_neu_ques_title'].values.reshape(-1,1))



# text_sent_neg_ques_title

# text_sent_pos_answer

# text_sent_pos_ques_body



# embeddings_train['num_words_ques_title'] = train['num_words_ques_title']

# embeddings_train['num_words_ques_body'] = train['num_words_ques_body']

# embeddings_train['num_words_answer'] = train['num_words_answer']



# embeddings_test['num_words_ques_title'] = np.vstack(test['num_words_ques_title'])

# embeddings_test['num_words_ques_body'] = np.vstack(test['num_words_ques_body'])

# embeddings_test['num_words_answer'] = np.vstack(test['num_words_answer'])





# # num of unique words in text

# embeddings_train['num_uniq_words_ques_title'] =  np.vstack(train['num_uniq_words_ques_title'])

# embeddings_train['num_uniq_words_ques_body'] = np.vstack(train['num_uniq_words_ques_body'])

# embeddings_train['num_uniq_words_answer'] = np.vstack(train['num_uniq_words_answer'])



# embeddings_test['num_uniq_words_ques_title'] = np.vstack(test['num_uniq_words_ques_title'])

# embeddings_test['num_uniq_words_ques_body'] = np.vstack(test['num_uniq_words_ques_body'])

# embeddings_test['num_uniq_words_answer'] = np.vstack(test['num_uniq_words_answer'])





# # num of characters in text

# embeddings_train['num_chars_ques_title'] =  np.vstack(train['num_chars_ques_title'])

# embeddings_train['num_chars_ques_body'] = np.vstack(train['num_chars_ques_body'])

# embeddings_train['num_chars_answer'] = np.vstack(train['num_chars_answer'])



# embeddings_test['num_chars_ques_title'] = np.vstack(test['num_chars_ques_title'])

# embeddings_test['num_chars_ques_body'] = np.vstack(test['num_chars_ques_body'])

# embeddings_test['num_chars_answer'] = np.vstack(test['num_chars_answer'])





# # num of stop_words in text

# embeddings_train['num_stop_words_ques_title'] =  np.vstack(train['num_stop_words_ques_title'])

# embeddings_train['num_stop_words_ques_body'] = np.vstack(train['num_stop_words_ques_body'])

# embeddings_train['num_stop_words_answer'] = np.vstack(train['num_stop_words_answer'])



# embeddings_test['num_stop_words_ques_title'] = np.vstack(test['num_stop_words_ques_title'])

# embeddings_test['num_stop_words_ques_body'] = np.vstack(test['num_stop_words_ques_body'])

# embeddings_test['num_stop_words_answer'] = np.vstack(test['num_stop_words_answer'])





# # num of punctuations in text

# embeddings_train['num_puncts_ques_title'] =  np.vstack(train['num_puncts_ques_title'])

# embeddings_train['num_puncts_ques_body'] = np.vstack(train['num_puncts_ques_body'])

# embeddings_train['num_puncts_answer'] = np.vstack(train['num_puncts_answer'])



# embeddings_test['num_puncts_ques_title'] = np.vstack(test['num_puncts_ques_title'])

# embeddings_test['num_puncts_ques_body'] = np.vstack(test['num_puncts_ques_body'])

# embeddings_test['num_puncts_answer'] = np.vstack(test['num_puncts_answer'])







# # Average length of words in text

# embeddings_train['mean_len_words_ques_title'] =  np.vstack(train['mean_len_words_ques_title'])

# embeddings_train['mean_len_words_ques_body'] = np.vstack(train['mean_len_words_ques_body'])

# embeddings_train['mean_len_words_ques_answer'] = np.vstack(train['mean_len_words_ques_answer'])



# embeddings_test['mean_len_words_ques_title'] = np.vstack(test['mean_len_words_ques_title'])

# embeddings_test['mean_len_words_ques_body'] = np.vstack(test['mean_len_words_ques_body'])

# embeddings_test['mean_len_words_ques_answer'] = np.vstack(test['mean_len_words_ques_answer'])





# # pos sentiment

# embeddings_train['text_sent_pos_ques_title'] =  np.vstack(train['text_sent_pos_ques_title'])

# embeddings_train['text_sent_pos_ques_body'] = np.vstack(train['text_sent_pos_ques_body'])

# embeddings_train['text_sent_pos_answer'] = np.vstack(train['text_sent_pos_answer'])



# embeddings_test['text_sent_pos_ques_title'] = np.vstack(test['text_sent_pos_ques_title'])

# embeddings_test['text_sent_pos_ques_body'] = np.vstack(test['text_sent_pos_ques_body'])

# embeddings_test['text_sent_pos_answer'] = np.vstack(test['text_sent_pos_answer'])



# # neg sentiment

# embeddings_train['text_sent_neg_ques_title'] =  np.vstack(train['text_sent_neg_ques_title'])

# embeddings_train['text_sent_neg_ques_body'] = np.vstack(train['text_sent_neg_ques_body'])

# embeddings_train['text_sent_neg_answer'] = np.vstack(train['text_sent_neg_answer'])



# embeddings_test['text_sent_neg_ques_title'] = np.vstack(test['text_sent_neg_ques_title'])

# embeddings_test['text_sent_neg_ques_body'] = np.vstack(test['text_sent_neg_ques_body'])

# embeddings_test['text_sent_neg_answer'] = np.vstack(test['text_sent_neg_answer'])



# # neu sentiment

# embeddings_train['text_sent_neu_ques_title'] =  np.vstack(train['text_sent_neu_ques_title'])

# embeddings_train['text_sent_neu_ques_body'] = np.vstack(train['text_sent_neu_ques_body'])

# embeddings_train['text_sent_neu_answer'] = np.vstack(train['text_sent_neu_answer'])



# embeddings_test['text_sent_neu_ques_title'] = np.vstack(test['text_sent_neu_ques_title'])

# embeddings_test['text_sent_neu_ques_body'] = np.vstack(test['text_sent_neu_ques_body'])

# embeddings_test['text_sent_neu_answer'] = np.vstack(test['text_sent_neu_answer'])

embeddings_train
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)



dist_features_train = np.array([

    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),

    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),

    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),

    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),

    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),

    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),

]).T



dist_features_test = np.array([

    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),

    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),

    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),

    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),

    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),

    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])

]).T



X_train = np.hstack([item for k, item in embeddings_train.items()] + [features_train, dist_features_train])

X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])

y_train = train[targets].values
# Compatible with tensorflow backend

class SpearmanRhoCallback(Callback):

    def __init__(self, training_data, validation_data, patience, model_name):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]

        

        self.patience = patience

        self.value = -1

        self.bad_epochs = 0

        self.model_name = model_name



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.x_val)

        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])

        if rho_val >= self.value:

            self.value = rho_val

            self.model.save_weights(self.model_name)

        else:

            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:

            print("Epoch %05d: early stopping Threshold" % epoch)

            self.model.stop_training = True

        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')

        return rho_val



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
from keras.layers.normalization import BatchNormalization



# def create_model():

#     inps = Input(shape=(X_train.shape[1],))

    

#     x = Dense(512, activation='relu')(inps)

#     x = Dropout(0.5)(x)

# #     x = Dense(256, activation='relu')(x) 

# #     x = Dropout(rate=0.5)(x)

#     x = Dense(128, activation='relu')(x)

# #     x = Dropout(0.5)(x)

#     x = BatchNormalization()(x)

#     x = Dense(y_train.shape[1], activation='sigmoid')(x)

#     model = Model(inputs=inps, outputs=x)

#     model.compile(

#         optimizer=Adam(lr=1e-4),

#         loss=['binary_crossentropy']

#     )

#     model.summary()

#     return model





def create_model():

    inps = Input(shape=(X_train.shape[1],))

    x = Dense(512, activation='relu')(inps) 

    x = Dropout(rate=0.5)(x)

    x = Dense(y_train.shape[1], activation='sigmoid')(x)

    model = Model(inputs=inps, outputs=x)

    model.compile(

        optimizer=Adam(lr=1e-4),

        loss=['binary_crossentropy']

    )

    model.summary()

    return model
all_predictions = []



kf = KFold(n_splits=5, random_state=42, shuffle=True)

for ind, (tr, val) in enumerate(kf.split(X_train)):

    X_tr = X_train[tr]

    y_tr = y_train[tr]

    X_vl = X_train[val]

    y_vl = y_train[val]

    

    model = create_model()

    model.fit(

        X_tr, y_tr, epochs=150, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 

        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),

                                       patience=5, model_name=u'best_model_batch.h5')]

    )

    model.load_weights('best_model_batch.h5')

    all_predictions.append(model.predict(X_test))

    

    os.remove('best_model_batch.h5')

    

model = create_model()

model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=False)

all_predictions.append(model.predict(X_test))

    

kf = KFold(n_splits=5, random_state=2019, shuffle=True)

for ind, (tr, val) in enumerate(kf.split(X_train)):

    X_tr = X_train[tr]

    y_tr = y_train[tr]

    X_vl = X_train[val]

    y_vl = y_train[val]

    

    model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)

    model.fit(X_tr, y_tr)

    all_predictions.append(model.predict(X_test))

    

model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)

model.fit(X_train, y_train)

all_predictions.append(model.predict(X_test))
all_predictions
test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)

max_val = test_preds.max() + 1

test_preds = test_preds/max_val + 1e-12
for column_ind in range(y_train.shape[1]):

    curr_column = y_train[:, column_ind]

    values = np.unique(curr_column)

    map_quantiles = []

    for val in values:

        occurrence = np.mean(curr_column == val)

        cummulative = sum(el['occurrence'] for el in map_quantiles)

        map_quantiles.append({'value': val, 'occurrence': occurrence, 'cummulative': cummulative})

            

    for quant in map_quantiles:

        pred_col = test_preds[:, column_ind]

        q1, q2 = np.quantile(pred_col, quant['cummulative']), np.quantile(pred_col, min(quant['cummulative'] + quant['occurrence'], 1))

        pred_col[(pred_col >= q1) & (pred_col <= q2)] = quant['value']

        test_preds[:, column_ind] = pred_col
submission = pd.read_csv(path_join(data_dir, 'sample_submission.csv'))

submission[targets] = test_preds

submission.to_csv("submission.csv", index = False)

submission.head()