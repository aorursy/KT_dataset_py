import numpy as np
import pandas as pd
from math import ceil, floor
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.initializers import TruncatedNormal
from sklearn import model_selection
from transformers import BertConfig, TFBertPreTrainedModel, TFBertMainLayer, TFBertModel, BertTokenizer, PreTrainedTokenizer, AutoTokenizer
# from transformers.tokenizers import PreTrainedTokenizerFast
from tokenizers import BertWordPieceTokenizer

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
    
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options(
    {"auto_mixed_precision": True})
# read csv files
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
train_df.dropna(inplace=True)
train_df = train_df.drop(train_df[train_df.sentiment=='neutral'].index, axis=0).reset_index(drop=True)

test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
test_df.loc[:, "selected_text"] = test_df.text.values

submission_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

print("train shape =", train_df.shape)
print("test shape  =", test_df.shape)

# set some global variables
PATH = "../input/bert-base-uncased/"
MAX_SEQUENCE_LENGTH = 128
tokenizer = BertWordPieceTokenizer(f"{PATH}/vocab.txt", lowercase=True, add_special_tokens=False)
MODELNAME = 'bert'

# let's take a look at the data
train_df.head(10)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

!pip install twython

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys, gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import string
import time
import re
import nltk
import spacy
from spacy import displacy
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob


nlp = spacy.load("en_core_web_sm")
# txt = 'text soooooo is,typing.......'
# tags = [(token.pos_, token.dep_, token.idx, token.text) for token in nlp(txt) if token.pos_ != 'SPACE'] + [(..., ..., len(txt))]
# enc = tokenizer.encode(txt)
# tokens = enc.tokens
# offsets = enc.offsets
# pos_tags = []
# dep_tags = []
# index0 = 0
# # print(tags)
# print(tokens)
# for index, token in enumerate(tokens):
# #     print(index, offsets[index][0], tags[index0])
#     if offsets[index][0] == tags[index0+1][2]:
#         index0 += 1
#     pos_tags.append(tags[index0][0])
#     dep_tags.append(tags[index0][1])
# print(pos_tags, dep_tags)

sentiment_map = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699,
    }


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = re.sub('https?://\S+|www\.\S+', '', text) # remove any links
    text = re.sub('\s\s+', ' ', text) # remove any links
    return text.lower().strip()


def tag_bert_tokens(train, columns):
    for column in columns:
        attr_values = []
        for ind, (tags, bert_tokens_words_bpe_length) in enumerate(train[[column, 'bert_tokens_words_bpe_length']].values):
            if type(tags) is list:
                try:
                    if 'polarity' in column:
                        attr_values.append([[0]*len(tags[0])]*3 + [tags[index] for index, value in enumerate(bert_tokens_words_bpe_length) for _ in range(value)] + [[0]*len(tags[0])])
                    else:
                        attr_values.append([0]*3 + [tags[index] for index, value in enumerate(bert_tokens_words_bpe_length) for _ in range(value)] + [0])
                except:
                    print(train.iloc[ind].values)
                    print(len(bert_tokens_words_bpe_length), len(tags), tags, bert_tokens_words_bpe_length)
                    assert False
            else:
                attr_values.append([[0]]*3 + [[tags] for value in bert_tokens_words_bpe_length for _ in range(value)] + [[0]])
        train[column] = attr_values
    
def onehot_encode(sequence, set_of_tags, tag2idx):
    ohe = np.zeros((len(sequence), len(set_of_tags)))
    for index, tag in enumerate(sequence):
        if tag in set_of_tags:
            ohe[index, tag2idx[tag]] = 1
    return ohe

def label_encode(sequence, set_of_tags, tag2idx):
    encoded_seq = []
    for index, tag in enumerate(sequence):
        if tag in set_of_tags:
            encoded_seq.append(tag2idx[tag])
    return np.array(encoded_seq, dtype=np.int)

def encode_all(tagged_text, set_of_tags, tag2idx, encode_func=onehot_encode):
    sequences = []
    for sequence in tagged_text:
        sequences.append(encode_func(sequence, set_of_tags, tag2idx))
    return sequences


class SentimentAnalyzer:
    
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        if type(self.sentiment_analyzer) not in [SentimentIntensityAnalyzer, TextBlob]:
            raise BaseException('Unknown sentiment analyzer', self.sentiment_analyzer)
        
    def __call__(self, text):
        if isinstance(self.sentiment_analyzer, SentimentIntensityAnalyzer):
            return self.sentiment_analyzer.polarity_scores(text)['compound']
        elif isinstance(self.sentiment_analyzer, TextBlob):
            return TextBlob(text).sentiment.polarity
            

def text2tag(text, sentiment, tokens, offsets, data_model_config, n_gram_window_range=(0, 4), sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer()), 
                                                                                            SentimentAnalyzer(TextBlob('123'))]):
    if not data_model_config['pos_dep_tags'] and not data_model_config['sentiment_calculation']:
        return [], [], []
    if data_model_config['pos_dep_tags']:
        tags = [(token.pos_, token.dep_, token.idx, token.text) for token in nlp(text) if token.pos_ != 'SPACE'] + [(..., ..., len(text), '')]
        if data_model_config['sentiment_calculation']:
            spacy_text = ' '.join(list(zip(*tags[:-1]))[-1])
            polarity_scores = calculate_texts_polarity_score(spacy_text, sentiment, n_gram_window_range, sentiment_analyzers)
    else:
        tags = []
        spacy_text = ''
        for idx, char in enumerate(text):
            if char == ' ':
                continue
            elif char in string.punctuation or not tags or tags[-1][-1] in string.punctuation:
                tags.append(('', '', idx, char))
            else:
                tags[-1] = tags[-1][:3] + (tags[-1][3] + char,)
#         print(tags)
        tags += [(..., ..., len(text), '')]
                    
        spacy_text = ' '.join(list(zip(*tags[:-1]))[-1])
        polarity_scores = calculate_texts_polarity_score(spacy_text, sentiment, n_gram_window_range, sentiment_analyzers)
    pos_tags = []
    dep_tags = []
    polarity_scores_tokens = []
    index0 = 0
    # print(tags)
    for index, token in enumerate(tokens[3:-1]):
        index += 3
    #     print(index, offsets[index][0], tags[index0])
        if offsets[index][0] == tags[index0+1][2]:
            index0 += 1
        if data_model_config['pos_dep_tags']:
            pos_tags.append(tags[index0][0])
            dep_tags.append(tags[index0][1])
        if data_model_config['sentiment_calculation']:
            polarity_scores_tokens.append([])
            for ind, pol_token_scores in enumerate(polarity_scores):
                polarity_scores_tokens[-1] += polarity_scores[ind][index0]
    return pos_tags, dep_tags, polarity_scores_tokens

def words_polarity_score(words, sentiment, n_gram_window_range=(0, 4), sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer()), 
                                                                                            SentimentAnalyzer(TextBlob('123'))]):
    """
    n_gram_window: word[i - n_gram_window_range[1]], ..., words[i], ..., word[i + n_gram_window_range[1]]
    """
    sentiment_kef = 1 if sentiment == 'positive' else -1
    polarity_scores = []
    for _ in range(len(sentiment_analyzers)):
        polarity_scores.append([])
    for index, word in enumerate(words):
        for _ in range(len(sentiment_analyzers)):
            polarity_scores[_].append([])
        for window in range(*n_gram_window_range):
            for index_, sentiment_analyzer in enumerate(sentiment_analyzers):
                polarity_scores[index_][-1].append(sentiment_analyzer(' '.join(words[max(0, index-window):min(len(words), index+window+1)])))
    return polarity_scores

def calculate_texts_polarity_score(text, sentiment, n_gram_window_range=(0, 4), sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer()), 
                                                                                            SentimentAnalyzer(TextBlob('123'))]):
    
    column_basename = 'words_polarity_scores'
    polarity_scores = []
#     for _ in range(len(sentiment_analyzers)):
#         polarity_scores.append([])
        
    words = text.split()
    scores = words_polarity_score(words, sentiment, n_gram_window_range, sentiment_analyzers)
    for index, scores_ in enumerate(scores):
        polarity_scores.append(scores_)
            
#     for index, sentiment_analyzer in enumerate(sentiment_analyzers):
#         if isinstance(sentiment_analyzer.sentiment_analyzer, SentimentIntensityAnalyzer):
#             train[f'vader_{column_basename}'] = polarity_scores[index]
#         elif isinstance(sentiment_analyzer.sentiment_analyzer, TextBlob):
#             train[f'textblob_{column_basename}'] = polarity_scores[index]
# #         elif # add your own sentiment analyzer
    return polarity_scores

def count_sentences(text):
    return len(re.split('[.!?]+', text))

def concat_encoded_features(train, columns):
    concateneted_features = []
    for row in train[columns].values:
        concateneted_features.append(np.concatenate(row, axis=1))
    return concateneted_features

class Preprocessor:
    
    def __init__(self, data_model_config, sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer())]):
        self.data_model_config = data_model_config
        self.max_num_sents = 0
        self.max_num_tokens = 0
        self.pos2idx = {}
        self.dep2idx = {}
        self.sentiment_analyzers = sentiment_analyzers
        
        self.build_output_types()
    
    def preprocess(self, text, selected_text, sentiment, isTrain):
        text = text.decode('utf-8').strip()
        selected_text = selected_text.decode('utf-8').strip()
        sentiment = sentiment.decode('utf-8').strip()
#         print(text)
    #     X.fillna('', inplace=True)
#         if sentiment == 'neutral':
#             return ()
        # clean texts
        text = clean_text(text)
        
        # tokenize with offsets
        enc = tokenizer.encode(text)
        input_ids, offsets = enc.ids, enc.offsets
        
        if isTrain:
            selected_text = clean_text(selected_text)
#             print(selected_text)

            # find the intersection between text and selected text
            idx_start, idx_end = None, None
            for index in (i for i, c in enumerate(text) if c == selected_text[0]):
                if text[index:index+len(selected_text)] == selected_text:
                    idx_start = index
                    idx_end = index + len(selected_text)
                    break
#             print(idx_start, idx_end)

            intersection = [0] * len(text)
            if idx_start != None and idx_end != None:
                for char_idx in range(idx_start, idx_end):
                    intersection[char_idx] = 1

                # compute targets
                target_idx = []
                for i, (o1, o2) in enumerate(offsets):
                    if sum(intersection[o1: o2]) > 0: # label
                        target_idx.append(i)
    #                 if sum(intersection[o1: o2]) == (o2 - o1):
    #                     target_idx.append(i)

                target_start = target_idx[0] + 3
                target_end = target_idx[-1] + 3
            else:
                target_start = 0
                target_end = 0
        else:
            target_start = 0
            target_end = 0

        input_ids = [101, sentiment_map[sentiment], 102] + input_ids + [102]
        input_type_ids = [0, 0, 0] + [1] * (len(input_ids) - 3)
        attention_mask = [1] * len(input_ids)
        offsets = [(0, 0), (0, 0), (0, 0)] + offsets + [(0, 0)]

        if self.data_model_config['global_statistics']:
            # calculate number of sentences
            text_num_sents = count_sentences(text)
            # calculate number of tokens (words + punctuation)
            text_num_tokens = len(text.split(' '))

            # min_max scaling
            if isTrain:
                self.max_num_sents = max(self.max_num_sents, text_num_sents)
                self.max_num_tokens = max(self.max_num_tokens, text_num_tokens)
            text_num_sents = MAX_SEQUENCE_LENGTH * [[text_num_sents]]
            text_num_tokens = MAX_SEQUENCE_LENGTH * [[text_num_tokens]]
        
        # get pos, dep tags and calculate polarity scores of each word
        pos_tags, dep_tags, polarity_scores_tokens = text2tag(text, sentiment, input_ids, offsets, self.data_model_config, sentiment_analyzers=self.sentiment_analyzers)
        
#         print(polarity_scores_tokens)
        if self.data_model_config['pos_dep_tags']:
            for index, tag in enumerate(pos_tags):
                if tag not in self.pos2idx and isTrain:
    #                 print('new pos tag', tag)
                    self.pos2idx[tag] = len(self.pos2idx)# + 1
                pos_tags[index] = self.pos2idx.get(tag, len(self.pos2idx))
    #         pos_tags = [0, 0, 0] + pos_tags + [0]

            for index, tag in enumerate(dep_tags):
                if tag not in self.dep2idx and isTrain:
    #                 print('new dep tag', tag)
                    self.dep2idx[tag] = len(self.dep2idx)# + 1
                dep_tags[index] = self.dep2idx.get(tag, len(self.dep2idx))
    #         dep_tags = [0, 0, 0] + dep_tags + [0]
            pos_tag_indices, pos_tag_values, pos_tag_dense_shape = (np.array([np.arange(3, 3 + len(pos_tags)), pos_tags]).T,
                                                                    np.ones((len(pos_tags),)),
                                                                    (MAX_SEQUENCE_LENGTH, len(self.pos2idx)))
            dep_tag_indices, dep_tag_values, dep_tag_dense_shape = (np.array([np.arange(3, 3 + len(dep_tags)), dep_tags]).T,
                                                                    np.ones((len(dep_tags),)),
                                                                    (MAX_SEQUENCE_LENGTH, len(self.dep2idx)))
                

        padding_length = MAX_SEQUENCE_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            input_type_ids += [0] * padding_length
            offsets += [(0, 0)] * padding_length
            
            if self.data_model_config['sentiment_calculation']:
                polarity_scores_tokens = (np.zeros((3, len(polarity_scores_tokens[0]),)).tolist() + 
                                          polarity_scores_tokens + 
                                          (padding_length + 1) * np.zeros((1, len(polarity_scores_tokens[0]),)).tolist())
        elif padding_length < 0:
            # not yet implemented
            # truncates if input length > max_seq_len
            pass
    
#         print(pos_tags)
#         print('preproc')
        input_sample = (input_ids, attention_mask, input_type_ids,)
        if self.data_model_config['global_statistics']:
            input_sample += (text_num_sents, text_num_tokens,)
        if self.data_model_config['pos_dep_tags']:
            input_sample += ((pos_tag_indices, pos_tag_values, pos_tag_dense_shape), 
                             (dep_tag_indices, dep_tag_values, dep_tag_dense_shape),)
        if self.data_model_config['sentiment_calculation']:
            input_sample += (polarity_scores_tokens,)
#         print(len(input_sample))
        
        return (
            input_sample,
            (target_start, target_end),
            (text, offsets, selected_text, sentiment)
        )
                  
    def map_fn(self, *sample):
#         print('map')
        (
            input_sample,
            (target_start, target_end),
            (text, offsets, selected_text, sentiment)
        ) = sample
        
        if self.data_model_config['global_statistics']:
            text_num_sents, text_num_tokens = input_sample[3:5]
            text_num_sents /= self.max_num_sents
            text_num_tokens /= self.max_num_tokens
            input_sample = input_sample[:3] + (text_num_sents, text_num_tokens,) + input_sample[5:]
        if self.data_model_config['pos_dep_tags']:
            if self.data_model_config['sentiment_calculation']:
                ((pos_tag_indices, pos_tag_values, pos_tag_dense_shape), 
                (dep_tag_indices, dep_tag_values, dep_tag_dense_shape)) = input_sample[-3:-1]
            else:
                ((pos_tag_indices, pos_tag_values, pos_tag_dense_shape), 
                (dep_tag_indices, dep_tag_values, dep_tag_dense_shape)) = input_sample[-2:]
            pos_tags = tf.SparseTensor(indices=pos_tag_indices, values=pos_tag_values, dense_shape=(MAX_SEQUENCE_LENGTH, len(self.pos2idx)+1))
            dep_tags = tf.SparseTensor(indices=dep_tag_indices, values=dep_tag_values, dense_shape=(MAX_SEQUENCE_LENGTH, len(self.dep2idx)+1))
            if self.data_model_config['sentiment_calculation']:
                input_sample = input_sample[:-3] + (pos_tags, dep_tags,) + input_sample[-1:]
            else:
                input_sample = input_sample[:-2] + (pos_tags, dep_tags,)
        
        return (
            input_sample,
            (target_start, target_end),
            (text, offsets, selected_text, sentiment)
        )
    
    def build_output_types(self):
        input_types = (tf.dtypes.int32,  tf.dtypes.int32,   tf.dtypes.int32,) # input_ids, attention_ids, input_type_ids
        if self.data_model_config['global_statistics']:
            input_types += (tf.dtypes.int32,   tf.dtypes.int32,) # text_num_sents, text_num_tokens
        if self.data_model_config['pos_dep_tags']:
            input_types += ((tf.dtypes.int64,  tf.dtypes.float32,   tf.dtypes.int64),  # pos tags indices, values, shape
                            (tf.dtypes.int64,  tf.dtypes.float32,   tf.dtypes.int64),) # dep tags indices, values, shape
        if self.data_model_config['sentiment_calculation']:
            input_types += (tf.dtypes.float32,) # polarity scores
            
        self.OUTPUT_TYPES = (
            input_types,
            (tf.dtypes.int32,  tf.dtypes.int32), # target start, target end
            (tf.dtypes.string, tf.dtypes.int32, tf.dtypes.string, tf.dtypes.string) # text, offsets, selected_text, sentiment
        )
    
    
#     OUTPUT_SHAPES = (
#         (
#             (128,), (128,), (128,), 
#             (128, 1), (128, 1), ((None, 2), (None,), (2)), ((None, 2), (None,), (2)), (128, None)
#         ),
#         (
#             (), ()
#         ),
#         (
#             (), (128, 2), (), ()
#         )
#     )
    
    # AutoGraph will automatically convert Python code to
    # Tensorflow graph code. You could also wrap 'preprocess' 
    # in tf.py_function(..) for arbitrary python code
    def _generator(self, tweet, selected_text, sentiment, isTrain):
        for tw, st, se in zip(tweet, selected_text, sentiment):
            outputs = self.preprocess(tw, st, se, isTrain)
            if (not any(outputs[1]) or se == 'neutral') and isTrain: # target_start and target_end == 0
                continue
            yield outputs
    
    # This dataset object will return a generator
    def _create_dataset(self, *args):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=self.OUTPUT_TYPES,
#             output_shapes=cls.OUTPUT_SHAPES,
            args=args #tweet, selected_text, sentiment, isTrain
        )
    
    def create_dataset(self, dataframe, batch_size, isTrain, shuffle_buffer_size=-1):
        dataset = self._create_dataset(
            dataframe.text.values, 
            dataframe.selected_text.values, 
            dataframe.sentiment.values, 
            isTrain
        )#.map(Preprocessor.map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.data_model_config['global_statistics'] or self.data_model_config['pos_dep_tags']:
            for sample in dataset:
                pass

            dataset = dataset.map(self.map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            

        dataset = dataset.cache()
        if shuffle_buffer_size != -1:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
#         dataset = dataset
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

class BertQAModel(TFBertPreTrainedModel):
    
    DROPOUT_RATE = 0.1
    NUM_HIDDEN_STATES = 2
    
    def __init__(self, config, preprocessor, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.bert = TFBertMainLayer(config, name="bert")
        self.concat = L.Concatenate(axis=-1)
        self.dropout = L.Dropout(self.DROPOUT_RATE)
        self.fc = L.Dense(64, activation='relu')
        self.dropout1 = L.Dropout(self.DROPOUT_RATE)
        self.qa_outputs = L.Dense(
            config.num_labels, 
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            dtype='float32',
            name="qa_outputs")
        
        self.last_axis_dim = 0
        if preprocessor.data_model_config['global_statistics']:
            self.last_axis_dim += 2
        if preprocessor.data_model_config['pos_dep_tags']:
            self.last_axis_dim += len(preprocessor.pos2idx) + len(preprocessor.dep2idx) + 2
        if preprocessor.data_model_config['sentiment_calculation']:
            self.last_axis_dim += 4
        
    @tf.function
    def call(self, inputs, **kwargs):
        # outputs: Tuple[sequence, pooled, hidden_states]
        if type(inputs) is dict:
            last_out, pooled, hidden_states = self.bert(inputs, **kwargs)
        else:
            last_out, pooled, hidden_states = self.bert(*inputs[:3], **kwargs)
        
        hidden_states = self.dropout(last_out, training=kwargs.get("training", False))
#         hidden_states = last_out
        if self.last_axis_dim:
            if type(inputs) is not dict: # list or tuple
                hidden_states = self.concat([hidden_states, *inputs[3:]]) #tf.sparse.to_dense(inputs[3], default_value=0),tf.sparse.to_dense(inputs[4], default_value=0), 
            else:
                hidden_states = self.concat([hidden_states, tf.ones((3, 5, self.last_axis_dim))])
        hidden_states = self.fc(hidden_states)
        hidden_states = self.dropout1(hidden_states, training=kwargs.get("training", False))
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        
        return start_logits, end_logits
    
    
def train(model, dataset, loss_fn, optimizer):
    
    @tf.function
    def train_step(model, inputs, y_true, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
#             print(y_pred, y_true)
            start_loss  = loss_fn(y_true[0], y_pred[0])
            end_loss = loss_fn(y_true[1], y_pred[1])
            loss = start_loss + end_loss
#             scaled_loss = loss
            
            scaled_loss = optimizer.get_scaled_loss(loss)
    
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    epoch_loss = 0.
    for batch_num, sample in enumerate(dataset):
#         print(sample)
        loss, y_pred = train_step(
            model, sample[0], sample[1], loss_fn, optimizer)

        epoch_loss += loss

        print(
            f"training ... batch {batch_num+1:03d} : "
            f"train loss {epoch_loss/(batch_num+1):.3f} ",
            end='\r')
        
        
def predict(model, dataset, loss_fn, optimizer):
        
    def to_numpy(*args):
        out = []
        for arg in args:
            if arg.dtype == tf.string:
                arg = [s.decode('utf-8') for s in arg.numpy()]
                out.append(arg)
            else:
                arg = arg.numpy()
                out.append(arg)
        return out
    
    # Initialize accumulators
    offset = tf.zeros([0, 128, 2], dtype=tf.dtypes.int32)
    text = tf.zeros([0,], dtype=tf.dtypes.string)
    selected_text = tf.zeros([0,], dtype=tf.dtypes.string)
    sentiment = tf.zeros([0,], dtype=tf.dtypes.string)
    pred_start = tf.zeros([0, 128], dtype=tf.dtypes.float32)
    pred_end = tf.zeros([0, 128], dtype=tf.dtypes.float32)
    
    for batch_num, sample in enumerate(dataset):
        
        print(f"predicting ... batch {batch_num+1:03d}"+" "*20, end='\r')
        
        y_pred = model(sample[0], training=False)
        
        # add batch to accumulators
        pred_start = tf.concat((pred_start, y_pred[0]), axis=0)
        pred_end = tf.concat((pred_end, y_pred[1]), axis=0)
        offset = tf.concat((offset, sample[2][1]), axis=0)
        text = tf.concat((text, sample[2][0]), axis=0)
        selected_text = tf.concat((selected_text, sample[2][2]), axis=0)
        sentiment = tf.concat((sentiment, sample[2][3]), axis=0)

    # pred_start = tf.nn.softmax(pred_start)
    # pred_end = tf.nn.softmax(pred_end)
    
    pred_start, pred_end, text, selected_text, sentiment, offset = \
        to_numpy(pred_start, pred_end, text, selected_text, sentiment, offset)
    
    return pred_start, pred_end, text, selected_text, sentiment, offset


def decode_prediction(pred_start, pred_end, text, offset, sentiment):
    
    def decode(pred_start, pred_end, text, offset):

        decoded_text = ""
        for i in range(pred_start, pred_end+1):
#             print(offset[i])
            decoded_text += text[offset[i][0]:offset[i][1]]
            if (i+1) < len(offset) and offset[i][1] < offset[i+1][0]:
                decoded_text += " "
        return decoded_text
    
    decoded_predictions = []
#     print(pred_start.shape, pred_end.shape, offset.shape, text)
    for i in range(len(text)):
        if sentiment[i] == "neutral" or len(text[i].split()) < 2:
            decoded_text = text[i]
        else:
            idx_start = np.argmax(pred_start[i])
            idx_end = np.argmax(pred_end[i])
#             print(idx_start, idx_end, offset[i].tolist())
            if idx_start > idx_end:
                idx_end = idx_start 
#             print(idx_start, idx_end, offset[i])
            decoded_text = decode(idx_start, idx_end, text[i], offset[i])
#             print(decoded_text)
#             decoded_text = str(decode(idx_start, idx_end, text[i], offset[i]))
            if len(decoded_text.strip()) == 0:
                decoded_text = text[i]
        decoded_predictions.append(decoded_text.strip())
    
    return decoded_predictions

def decode_prediction1(pred_start, pred_end, text, offset, sentiment):
    
    def decode(pred_start, pred_end, text, offset):

        decoded_text = ""
        for i in range(pred_start, pred_end+1):
#             print(offset[i])
            decoded_text += text[offset[i][0]:offset[i][1]]
            if (i+1) < len(offset) and offset[i][1] < offset[i+1][0]:
                decoded_text += " "
                
        # post add
        while (i+1) < len(offset) and offset[i][1] == offset[i+1][0] and offset[i+1][0]:
            decoded_text += text[offset[i+1][0]:offset[i+1][1]]
            i += 1
        
        # pre add
        i = pred_start
        while i - 1 > 0 and offset[i-1][1] == offset[i][0] and offset[i-1][0]:
            decoded_text = text[offset[i-1][0]:offset[i-1][1]] + decoded_text
            i -= 1
        
        return decoded_text
    
    decoded_predictions = []
#     print(pred_start.shape, pred_end.shape, offset.shape, text)
    for i in range(len(text)):
        if sentiment[i] == "neutral" or len(text[i].split()) < 2:
            decoded_text = text[i]
        else:
            idx_start = np.argmax(pred_start[i])
            idx_end = np.argmax(pred_end[i])
#             print(idx_start, idx_end, offset[i].tolist())
            if idx_start > idx_end:
                idx_end = idx_start 
#             print(idx_start, idx_end, offset[i])
            decoded_text = decode(idx_start, idx_end, text[i], offset[i])
#             print(decoded_text)
#             decoded_text = str(decode(idx_start, idx_end, text[i], offset[i]))
            if len(decoded_text.strip()) == 0:
                decoded_text = text[i]
        decoded_predictions.append(decoded_text.strip())
    
    return decoded_predictions

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def train(model, dataset, loss_fn, optimizer):
    
#     @tf.function
    def train_step(model, inputs, y_true, loss_fn, optimizer): # add weight according to jacard score start-end span matrix
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
#             print(y_pred, y_true)
            y_true = (tf.sparse.to_dense(tf.sparse.SparseTensor(tf.cast(tf.stack((np.arange(y_pred[0].shape[0]), y_true[0]), axis=1), tf.int64), 
                                                                np.ones(y_pred[0].shape[0], dtype=np.int64), 
                                                                dense_shape=y_pred[0].shape),
                                         default_value=0), 
                      tf.sparse.to_dense(tf.sparse.SparseTensor(tf.cast(tf.stack((np.arange(y_pred[1].shape[0]), y_true[1]), axis=1), tf.int64), 
                                                                np.ones(y_pred[1].shape[0], dtype=np.int64), 
                                                                dense_shape=y_pred[1].shape),
                                         default_value=0))
#             print(y_pred, y_true[0].shape)
            start_loss  = tf.nn.softmax(y_pred[0])
            end_loss = tf.nn.softmax(y_pred[1])
#             print(start_loss, end_loss, tf.expand_dims(start_loss, axis=-1))
#             print((tf.expand_dims(start_loss, axis=-1) + tf.expand_dims(end_loss, axis=-2)).shape)
            start_end_interaction = tf.reshape((tf.expand_dims(start_loss, axis=-1) * tf.expand_dims(end_loss, axis=-2)) \
                                               * tf.cast(tf.linalg.band_part(np.ones((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)), -1, 0), tf.float32), 
                                               (y_pred[0].shape[0], -1))
            loss = loss_fn(y_true[0], y_pred[0]) + \
                   loss_fn(y_true[1], y_pred[1]) + \
                   tf.reduce_sum(start_end_interaction)
#             print(loss.shape, loss_fn(y_true[0], y_pred[0]).shape, loss_fn(y_true[0], y_pred[0]))
            scaled_loss = optimizer.get_scaled_loss(loss)
#         assert False
    
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    epoch_loss = 0.
    for batch_num, sample in enumerate(dataset):
#         print(sample)
        loss, y_pred = train_step(
            model, sample[0], sample[1], loss_fn, optimizer)

        epoch_loss += loss

        print(
            f"training ... batch {batch_num+1:03d} : "
            f"train loss {epoch_loss/(batch_num+1):.3f} ",
            end='\r')
num_folds = 5
num_epochs = 3
batch_size = 32
learning_rate = 3e-5

dmc = {
    'global_statistics': False,
    'pos_dep_tags': False,
    'sentiment_calculation': False
}
preproc = Preprocessor(dmc)

optimizer = tf.keras.optimizers.Adam(learning_rate)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    optimizer, 'dynamic')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn1 = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

kfold = model_selection.KFold(
    n_splits=num_folds, shuffle=True, random_state=42)

# initialize test predictions
test_preds_start = np.zeros((len(test_df), 128), dtype=np.float32)
test_preds_end = np.zeros((len(test_df), 128), dtype=np.float32)

for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(train_df.text)):
    print("\nfold %02d" % (fold_num+1))
        
#     break
    train_dataset = preproc.create_dataset(
        train_df.iloc[train_idx], batch_size, isTrain=True, shuffle_buffer_size=2048)#.shuffle(2048).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = preproc.create_dataset(
        train_df.iloc[valid_idx], batch_size, isTrain=False, shuffle_buffer_size=-1)#.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    test_dataset = preproc.create_dataset(
        test_df, batch_size, isTrain=False, shuffle_buffer_size=-1)
    
    
    config = BertConfig(output_hidden_states=True, num_labels=2)
    BertQAModel.DROPOUT_RATE = 0.1
    BertQAModel.NUM_HIDDEN_STATES = 2
    model = BertQAModel.from_pretrained(PATH, config=config, preprocessor=preproc)
#     break
    for epoch_num in range(num_epochs):
        print("\nepoch %03d" % (epoch_num+1))
        
        # train for an epoch
        train(model, train_dataset, loss_fn1, optimizer)
        
        # predict validation set and compute jaccardian distances
        pred_start, pred_end, text, selected_text, sentiment, offset = \
            predict(model, valid_dataset, loss_fn1, optimizer)
        
        selected_text_pred = decode_prediction(
            pred_start, pred_end, text, offset, sentiment)
        jaccards = []
        for i in range(len(selected_text)):
            jaccards.append(
                jaccard(selected_text[i], selected_text_pred[i]))
        
        score = np.mean(jaccards)
        print(f"valid jaccard epoch {epoch_num+1:03d}: {score}"+" "*15)
        
        selected_text_pred = decode_prediction1(
            pred_start, pred_end, text, offset, sentiment)
        jaccards = []
        for i in range(len(selected_text)):
            jaccards.append(
                jaccard(selected_text[i], selected_text_pred[i]))
        
        score = np.mean(jaccards)
        print(f"valid jaccard epoch {epoch_num+1:03d}: pre-post-processing {score}"+" "*15)
        
#         evaldef(model, valid_dataset, valid_idx, new_train, batch_size=batch_size)
    break
        
#         if score > best_score:
#             best_score = score
#             # requires you to have 'fold-{fold_num}' folder in PATH:
#             # model.save_pretrained(PATH+f'fold-{fold_num}')
#             # or
#             # model.save_weights(PATH + f'fold-{fold_num}.h5')
            
#             # predict test set
#             test_pred_start, test_pred_end, test_text, _, test_sentiment, test_offset = \
#                 predict(model, test_dataset, loss_fn, optimizer)
    
#     # add epoch's best test preds to test preds arrays
#     test_preds_start += test_pred_start * 0.2
#     test_preds_end += test_pred_end * 0.2
    
#     # reset model, as well as session and graph (to avoid OOM issues?) 
#     session = tf.compat.v1.get_default_session()
#     graph = tf.compat.v1.get_default_graph()
#     del session, graph, model
#     model = BertQAModel.from_pretrained(PATH, config=config)
    
# # decode test set and add to submission file
# selected_text_pred = decode_prediction(
#     test_preds_start, test_preds_end, test_text, test_offset, test_sentiment)


# # Update 3 (see https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942)
# def f(selected):
#     return " ".join(set(selected.lower().split()))
# submission_df.loc[:, 'selected_text'] = selected_text_pred
# submission_df['selected_text'] = submission_df['selected_text'].map(f)

# submission_df.to_csv("submission.csv", index=False)
