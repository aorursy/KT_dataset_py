# importing necessary libraries

import random

import html



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import tensorflow as tf

import tensorflow.keras.backend as K

import os

from scipy.stats import spearmanr

from scipy.optimize import minimize

from transformers import *

from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model

from sklearn.model_selection import KFold

from scipy.stats import spearmanr
%matplotlib inline

from IPython.display import Image

Image('../input/google-qna-ensemble-architecture/architecture.png')
from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ["model","dataset","train loss","cv loss","train rhos","cv rhos"] 



x.add_row(['bert_base_uncased','questions',0.3393, 0.3302, 0.5543, 0.6013]) 

x.add_row(['bert_base_uncased','answer',0.3320, 0.3278, 0.4967, 0.5438]) 

x.add_row(['bert_base_uncased','question+answer',0.3287, 0.3166, 0.5511, 0.6109])



x.add_row(['roberta_base','questions',0.3542, 0.3400, 0.4953, 0.5674]) 

x.add_row(['roberta_base','answer',0.3430, 0.3253, 0.3927, 0.4993]) 

x.add_row(['roberta_base','question+answer',0.3546, 0.3397, 0.4305, 0.5082])



x.add_row(['xlnet_base_cased','questions',0.3662, 0.3412, 0.4679, 0.5685]) 

x.add_row(['xlnet_base_cased','answer',0.3611, 0.3401, 0.3531, 0.4702]) 

x.add_row(['xlnet_base_cased','question+answer',0.3721, 0.3452, 0.3942, 0.5013]) 

print(x)
import warnings 

warnings.filterwarnings('ignore')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# fixing random seeds

seed = 13

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

tf.random.set_seed(seed)
def get_data():

    print('getting test and train data...')

    # reading the data into dataframe using pandas

    path = '../input/google-quest-challenge/'

    train = pd.read_csv(path+'train.csv')

    test = pd.read_csv(path+'test.csv')

    submission = pd.read_csv(path+'sample_submission.csv')



    # Selecting data for training and testing

    y = train[train.columns[11:]] # storing the target values in y

    X = train[['question_title', 'question_body', 'answer']]

    X_test = test[['question_title', 'question_body', 'answer']]



    # Cleaning the data

    X.question_body = X.question_body.apply(html.unescape)

    X.question_title = X.question_title.apply(html.unescape)

    X.answer = X.answer.apply(html.unescape)



    X_test.question_body = X_test.question_body.apply(html.unescape)

    X_test.question_title = X_test.question_title.apply(html.unescape)

    X_test.answer = X_test.answer.apply(html.unescape)



    return X, X_test, y, train, test
def get_tokenizer(model_name):

    print(f'getting tokenizer for {model_name}...')

    if model_name == 'xlnet-base-cased':

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    elif model_name == 'roberta-base':

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    elif model_name == 'bert-base-uncased':

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    

    return tokenizer
def fix_length(tokens, max_sequence_length=512, q_max_len=254, a_max_len=254, model_type='questions'):

    if model_type == 'questions':

        length = len(tokens)

        if length > max_sequence_length:

            tokens = tokens[:max_sequence_length-1]

        return tokens



    else:

        question_tokens, answer_tokens = tokens

        q_len = len(question_tokens)

        a_len = len(answer_tokens)

        if q_len + a_len + 3 > max_sequence_length:

            if a_max_len <= a_len and q_max_len <= q_len:

                q_new_len_head = q_max_len//2

                question_tokens = question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]

                a_new_len_head = a_max_len//2

                answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]

            elif q_len <= a_len and q_len < q_max_len:

                a_max_len = a_max_len + (q_max_len - q_len - 1)

                a_new_len_head = a_max_len//2

                answer_tokens = answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]

            elif a_len < q_len:

                q_max_len = q_max_len + (a_max_len - a_len - 1)

                q_new_len_head = q_max_len//2

                question_tokens = question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]



    return question_tokens, answer_tokens
# function for tokenizing the input data for transformer.

def transformer_inputs(title, question, answer, tokenizer, model_type='questions', MAX_SEQUENCE_LENGTH = 512):

    if model_type == 'questions':

        question = f"{title} [SEP] {question}"

        question_tokens = tokenizer.tokenize(question)

        question_tokens = fix_length(question_tokens, model_type=model_type)

        ids_q = tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens)

        padded_ids = (ids_q + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[:MAX_SEQUENCE_LENGTH]

        token_type_ids = ([0] * MAX_SEQUENCE_LENGTH)[:MAX_SEQUENCE_LENGTH]

        attention_mask = ([1] * len(ids_q) + [0] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[:MAX_SEQUENCE_LENGTH]

        

        return padded_ids, token_type_ids, attention_mask



    else:

        question = f"{title} [SEP] {question}"

        question_tokens = tokenizer.tokenize(question)

        answer_tokens = tokenizer.tokenize(answer)

        question_tokens, answer_tokens = fix_length(tokens=(question_tokens, answer_tokens), model_type=model_type)

        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"] + answer_tokens + ["[SEP]"])

        padded_ids = ids + [tokenizer.pad_token_id] * (MAX_SEQUENCE_LENGTH - len(ids))

        token_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(answer_tokens) + 1) + [0] * (MAX_SEQUENCE_LENGTH - len(ids))

        attention_mask = [1] * len(ids) + [0] * (MAX_SEQUENCE_LENGTH - len(ids))



        return padded_ids, token_type_ids, attention_mask
# function for creating the input_ids, masks and segments for the bert input

def input_data(df, tokenizer, model_type='questions'):

    print(f'generating {model_type} input for transformer...')

    input_ids, input_token_type_ids, input_attention_masks = [], [], []

    for title, body, answer in tqdm(zip(df["question_title"].values, df["question_body"].values, df["answer"].values)):

        ids, type_ids, mask = transformer_inputs(title, body, answer, tokenizer, model_type=model_type)

        input_ids.append(ids)

        input_token_type_ids.append(type_ids)

        input_attention_masks.append(mask)

    

    return (

        np.asarray(input_ids, dtype=np.int32),

        np.asarray(input_attention_masks, dtype=np.int32),

        np.asarray(input_token_type_ids, dtype=np.int32))
def get_model(name):

    if name == 'xlnet-base-cased':

        config = XLNetConfig.from_pretrained('xlnet-base-cased', output_hidden_states=True)

        model = TFXLNetModel.from_pretrained('xlnet-base-cased', config=config)

    elif name == 'roberta-base':

        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)

        model = TFRobertaModel.from_pretrained('roberta-base', config=config)

    elif name == 'bert-base-uncased':

        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)

        model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    return model
def create_model(name='xlnet-base-cased', model_type='questions'):

    print(f'creating model {name}...')

    # Creating the model

    K.clear_session()

    max_seq_length = 512



    input_tokens = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_tokens")

    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

    input_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_segment")



    model = get_model(name)

    if (name == 'xlnet-base-cased'):

      sequence_output, hidden_states = model([input_tokens, input_mask])

    elif (name=='roberta-base' and model_type!='questions'):

      sequence_output, pooler_output, hidden_states = model([input_tokens, input_mask])

    else:

      sequence_output, pooler_output, hidden_states = model([input_tokens, input_mask, input_segment])



    # Last 4 hidden layers of bert

    h12 = tf.reshape(hidden_states[-1][:,0],(-1,1,768))

    h11 = tf.reshape(hidden_states[-2][:,0],(-1,1,768))

    h10 = tf.reshape(hidden_states[-3][:,0],(-1,1,768))

    h09 = tf.reshape(hidden_states[-4][:,0],(-1,1,768))

    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])



    x = GlobalAveragePooling1D()(concat_hidden)



    x = Dropout(0.2)(x)



    if model_type == 'answers':

      output = Dense(9, activation='sigmoid')(x)

    elif model_type == 'questions':

      output = Dense(21, activation='sigmoid')(x)

    else:

      output = Dense(30, activation='sigmoid')(x)



    if (name == 'xlnet-base-cased') or (name=='roberta-base' and model_type!='questions'):

      model = Model(inputs=[input_tokens, input_mask], outputs=output)

    else:

      model = Model(inputs=[input_tokens, input_mask, input_segment], outputs=output)



    return model
class data_generator:

  def __init__(self, X, X_test, tokenizer, type_):

      # test data

      tokens, masks, segments = input_data(X_test, tokenizer, type_)

      self.test_data = {'input_tokens': tokens, 

                        'input_mask': masks,

                        'input_segment': segments} 



      # Train data

      self.tokens, self.masks, self.segments = input_data(X, tokenizer, type_)

  def generate_data(tr, cv, name='xlnet-base-cased', model_type='questions'):

      if name!='xlnet-base-cased':

          train_data = {'input_tokens': self.tokens[tr], 

                        'input_mask': self.masks[tr],

                        'input_segment': self.segments[tr]}



          cv_data = {'input_tokens': self.tokens[cv], 

                    'input_mask': self.masks[cv],

                    'input_segment': self.segments[cv]}

      else:

          train_data = {'input_tokens': self.tokens[tr], 

                        'input_mask': self.masks[tr]}



          cv_data = {'input_tokens': self.tokens[cv], 

                    'input_mask': self.masks[cv]}



      if model_type=='questions':

          y_tr = y.values[tr, 21:]

          y_cv = y.values[cv, 21:]



      elif model_type=='answers':

          y_tr = y.values[tr, 21:]

          y_cv = y.values[cv, 21:]



      else:

          y_tr = y.values[tr]

          y_cv = y.values[cv]  



      return train_data, cv_data, y_tr, y_cv
# https://www.kaggle.com/markpeng/ensemble-5models-v4-v7-magic/notebook?select=submission.csv#Do-Inference

def optimize_ranks(preds, unique_labels):

    print(f'optimizing the predicted values...')

    new_preds = np.zeros(preds.shape) 

    for i in range(preds.shape[1]):

        interpolate_bins = np.digitize(preds[:, i], bins=unique_labels, right=False)

        if len(np.unique(interpolate_bins)) == 1: 

            new_preds[:, i] = preds[:, i]

        else:

            new_preds[:, i] = unique_labels[interpolate_bins]



    return new_preds
# https://www.kaggle.com/markpeng/ensemble-5models-v4-v7-magic/notebook?select=submission.csv#Do-Inference

def get_exp_labels(train):

    X = train.iloc[:, 11:]

    unique_labels = np.unique(X.values)

    denominator = 60

    q = np.arange(0, 101, 100 / denominator)

    exp_labels = np.percentile(unique_labels, q) # Generating the 60 bins.

    return exp_labels
# Function to calculate the Spearman's rank correlation coefficient 'rhos' of actual and predicted data.

def compute_spearmanr_ignore_nan(trues, preds):

    rhos = []

    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):

        rhos.append(spearmanr(tcol, pcol).correlation)

    return np.nanmean(rhos)
# Making the 'rhos' metric to tensorflow graph compatible.

def rhos(y, y_pred):

    return tf.py_function(compute_spearmanr_ignore_nan, (y, y_pred), tf.double)
def fit_model(model, model_name, model_type, data_gen, file_path, train, use_saved_weights=True): 

  path = '../input/google-qna-predicted-data/'

  if use_saved_weights:

    print(f'getting saved weights for {model_name}...')

    model.load_weights(path+file_path)



  else:

    print(f'fitting data on {model_name}...')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[rhos])

    kf = KFold(n_splits=5, random_state=42)

    for tr, cv in kf.split(np.arange(train.shape[0])):

      tr_data, cv_data, y_tr, y_cv = data_gen.generate_data(tr, cv, model_name, model_type)

      model.fit(tr_data, y_tr, epochs=1, batch_size=4, validation_data=(cv_data, y_cv))

      model.save_weights(file_path)



  return model
def get_weighted_avg(model_predictions):

  xlnet_q, xlnet_a, roberta_q, roberta_a, roberta_qa, bert_q, bert_a, bert_qa = model_predictions

  xlnet_concat = np.concatenate((xlnet_q, xlnet_a), axis=1)

  bert_concat = np.concatenate((bert_q, bert_a), axis=1)

  roberta_concat = np.concatenate((roberta_q, roberta_a), axis=1)

  predict = (roberta_qa + bert_qa + xlnet_concat + bert_concat + roberta_concat)/5



  return predict
def get_predictions(predictions_present=True, model_saved_weights_present=True):

  msw = model_saved_weights_present

  X, X_test, y, train, test = get_data()

  path = '../input/google-qna-predicted-data/'

  model_names = ['xlnet-base-cased', 'roberta-base', 'bert-base-uncased']

  model_types = ['questions', 'answers', 'questions_answers']

  saved_weights_names = ['xlnet_q.h5', 'xlnet_a.h5', 'roberta_q.h5', 'roberta_a.h5', 

                        'roberta_qa.h5', 'bert_q.h5', 'bert_a.h5', 'bert_qa.h5']



  saved_model_predictions = [path+'xlnet_q.csv', path+'xlnet_a.csv', path+'roberta_q.csv', path+'roberta_a.csv', 

                              path+'roberta_qa.csv', path+'bert_q.csv', path+'bert_a.csv', path+'bert_qa.csv']

  model_predictions = []



  if predictions_present:

    model_predictions = [pd.read_csv(file_name).values for file_name in saved_model_predictions]



  else:

    i = 0

    for name_ in model_names:

      for type_ in model_types:

        if name_ == 'xlnet-base-cased' and type_ == 'questions_answers':

          continue

        print('-'*100)

        model = create_model(name_, type_)

        tokenizer = get_tokenizer(name_)

        data_gen = data_generator(X, X_test, tokenizer, type_)

        model = fit_model(model, name_, type_, data_gen, saved_weights_names[i], train, msw)

        print(f'getting target predictions from {name_}...')

        model_predictions.append(model.predict(data_gen.test_data))

        i+=1



  predicted_labels = get_weighted_avg(model_predictions)

  exp_labels = get_exp_labels(train)

  optimized_predicted_labels = optimize_ranks(predicted_labels, exp_labels)

  df = pd.concat([test['qa_id'], pd.DataFrame(optimized_predicted_labels, columns=train.columns[11:])], axis=1)

  print('done...!')



  return df
submission = get_predictions(predictions_present=True)

# submission.to_csv('submission.csv')
submission = pd.read_csv('../input/google-qna-predicted-data/output.csv')

sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')



sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

id_in_sub = set(submission.qa_id)

id_in_sample_submission = set(sample_submission.qa_id)

diff = id_in_sample_submission - id_in_sub



sample_submission = pd.concat([submission, sample_submission[sample_submission.qa_id.isin(diff)]]).reset_index(drop=True)

sample_submission.to_csv("submission.csv", index=False)