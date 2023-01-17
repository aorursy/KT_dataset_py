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
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = re.sub('https?://\S+|www\.\S+', '', text) # remove any links
    text = re.sub('&quot;', '"', text)
    text = re.sub('&amp;', '&', text)
    # replace multiple non-latin and non-digit chars into one char. E.g.: '!!!!' into '!'
    text = re.sub(r'([^a-zA-Z0-9\s])\1+', r'\g<1>', text) 
    # e.g.: replace 'nothing.to.do' into 'nothing. to. do' 
    # in order to interpret them as different tokens for vader.sentimentAnalyzer
    text = re.sub(r'([?!.])', '\g<1> ', text) 
    return text.strip()

def text2postag(train, column, attributes):
    attr_values = [[] for _ in attributes]
    for text in train[column].values:
        for index, attr in enumerate(attributes):
            attr_values[index].append([])
        for token in nlp(text):
            if token.pos_ == 'SPACE':
                continue
            for index, attr in enumerate(attributes):
                attr_values[index][-1].append(getattr(token, attr))
    for index, attr in enumerate(attributes):
        train[f'{column}_{attr}'] = attr_values[index]

def label_sequences(train):
    target_sequences = []
    weights = []
    for index, sequence in enumerate(train.cleaned_text_text.values):
        selected_sequence = train.loc[index, 'cleaned_selected_text_text']
        labeled_sequence = np.zeros(len(sequence), dtype=np.int)
        if not selected_sequence:
            target_sequences.append(labeled_sequence)
            weights.append(labeled_sequence)
            continue
        local_index = 0
        while len(sequence) >= local_index + len(selected_sequence):
            if sequence[local_index : local_index+len(selected_sequence)] == selected_sequence:
                labeled_sequence[local_index : local_index+len(selected_sequence)] = 1
                target_sequences.append(labeled_sequence)
                weights.append(np.ones(len(sequence), dtype=np.float16))
                break
            local_index += 1
        else:
            local_index = 0
            while len(sequence) >= local_index + len(selected_sequence):
                if (sequence[local_index][-len(selected_sequence[0]):] == selected_sequence[0] and 
                    sequence[local_index + len(selected_sequence) - 1][:len(selected_sequence[-1])] == selected_sequence[-1] and 
                    sequence[local_index+1 : local_index+len(selected_sequence)-1] == selected_sequence[1:-1]):
                    
                    labeled_sequence[local_index : local_index+len(selected_sequence)] = 1
                    target_sequences.append(labeled_sequence)
                    weights.append(np.ones(len(sequence), dtype=np.float16))
                    if sequence[local_index] == selected_sequence[0]:
                        weights[-1][local_index] = 0.5
                    if sequence[local_index + len(selected_sequence) - 1] == selected_sequence[-1]:
                        weights[-1][local_index + len(selected_sequence) - 1] = 0.5
                    break
                local_index += 1
            else:
                target_sequences.append(labeled_sequence)
                weights.append(labeled_sequence)
    train['target_sequence'] = target_sequences
    train['target_sequence_weights'] = weights
    
def drop_unlabled_sequences(train):
    cnt = 0
    drop_indexes = []
    for index, seq in enumerate(train.target_sequence.values):
        if not seq.any():
            cnt += 1
            drop_indexes += [index]
    if drop_indexes:
        train.drop(drop_indexes, inplace=True, axis=0)
        train.reset_index(drop=True, inplace=True)
    print(f'dropped {cnt} samples')
    
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

def calculate_texts_polarity_score(train, n_gram_window_range=(0, 4), sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer()), 
                                                                                            SentimentAnalyzer(TextBlob('123'))]):
    
    column_basename = 'words_polarity_scores'
    polarity_scores = []
    for _ in range(len(sentiment_analyzers)):
        polarity_scores.append([])
        
    for words, sentiment in train[['cleaned_text_text', 'sentiment']].values:
        scores = words_polarity_score(words, sentiment, n_gram_window_range, sentiment_analyzers)
        for index, scores_ in enumerate(scores):
            polarity_scores[index].append(scores_)
            
    for index, sentiment_analyzer in enumerate(sentiment_analyzers):
        if isinstance(sentiment_analyzer.sentiment_analyzer, SentimentIntensityAnalyzer):
            train[f'vader_{column_basename}'] = polarity_scores[index]
        elif isinstance(sentiment_analyzer.sentiment_analyzer, TextBlob):
            train[f'textblob_{column_basename}'] = polarity_scores[index]
#         elif # add your own sentiment analyzer
    return

def count_sentences(text):
    return len(re.split('[.!?]+', text))

def concat_encoded_features(train, columns):
    concateneted_features = []
    for row in train[columns].values:
        concateneted_features.append(np.concatenate((*row[:-2], [[row[-2]]]*len(row[0]), [[row[-1]]]*len(row[0])), axis=1))
    return concateneted_features

def pad_sequence(feature_sequence, target_sequence, target_weight_sequence, max_seq_len, mode='uniform'):
    """
    mode: uniform - num of zero-padding for pre and post padding is aprxmtly the same
          pre - ...
          post - ...
    """
    if mode == 'uniform':
        pre_pad_length = (max_seq_len - len(feature_sequence)) // 2
        pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
        post_pad_length = max_seq_len - len(feature_sequence) - pre_pad_length
        post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
        feature_sequence = (feature_sequence,)
        target_sequence = (target_sequence,)
        target_weight_sequence = (target_weight_sequence,)
        if pre_pad:
            feature_sequence = (pre_pad,) + feature_sequence
            target_sequence = ([0]*pre_pad_length,) + target_sequence
            target_weight_sequence = ([1]*pre_pad_length,) + target_weight_sequence
        if post_pad:
            feature_sequence += (post_pad,)
            target_sequence += ([0]*post_pad_length,)
            target_weight_sequence += ([1]*post_pad_length,)
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
            target_sequence = np.concatenate(target_sequence, axis=0)
            target_weight_sequence = np.concatenate(target_weight_sequence, axis=0)
        else:
            feature_sequence, target_sequence, target_weight_sequence = feature_sequence[0], target_sequence[0], target_weight_sequence[0]
    elif mode == 'pre':
        pre_pad_length = (max_seq_len - len(feature_sequence)) // 2
        pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
        feature_sequence = (feature_sequence,)
        target_sequence = (target_sequence,)
        target_weight_sequence = (target_weight_sequence,)
        if pre_pad:
            feature_sequence = (pre_pad,) + feature_sequence
            target_sequence = ([0]*pre_pad_length,) + target_sequence
            target_weight_sequence = ([1]*pre_pad_length,) + target_weight_sequence
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
            target_sequence = np.concatenate(target_sequence, axis=0)
            target_weight_sequence = np.concatenate(target_weight_sequence, axis=0)
        else:
            feature_sequence, target_sequence, target_weight_sequence = feature_sequence[0], target_sequence[0], target_weight_sequence[0]
    elif mode == 'post':
        post_pad_length = (max_seq_len - len(feature_sequence)) // 2
        post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
        feature_sequence = (feature_sequence,)
        target_sequence = (target_sequence,)
        target_weight_sequence = (target_weight_sequence,)
        if post_pad:
            feature_sequence += (post_pad,)
            target_sequence += ([0]*post_pad_length,)
            target_weight_sequence += ([1]*post_pad_length,)
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
            target_sequence = np.concatenate(target_sequence, axis=0)
            target_weight_sequence = np.concatenate(target_weight_sequence, axis=0)
        else:
            feature_sequence, target_sequence, target_weight_sequence = feature_sequence[0], target_sequence[0], target_weight_sequence[0]
    else:
        raise BaseException(f'Unknown mode: {mode}. Choose one of the following: ["uniform","pre", "post"]')
    return feature_sequence, target_sequence, target_weight_sequence

def pad_sequences(features, target, target_weights, max_seq_len, mode='uniform'):
    features, target, target_weights = list(zip(*[pad_sequence(features[index], target[index], target_weights[index], max_seq_len, mode) \
                                               for index in range(len(features))]))
    return features, target, target_weights



class Preprocessor:

    def __init__(self, cat_encoding_func, sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer())]):
        self.isTrain = True
        self.cat_encoding_func = cat_encoding_func
        self.sentiment_analyzers = sentiment_analyzers
    
    def fit(self, X, y=None):
        self.isTrain = True
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.isTrain:
            X.fillna('', inplace=True)
        # drop neutral samples, because we would select the whole text for neutral sentiment text
        X.drop(X[X.sentiment == 'neutral'].index, inplace=True)
        X.reset_index(drop=True, inplace=True)
        # clean texts
        print('Clean text...')
        X['cleaned_text'] = X.text.apply(clean_text)
        print('text2postag...')
        text2postag(X, 'cleaned_text', ['pos_', 'dep_', 'text'])
        if self.isTrain:
            X['cleaned_selected_text'] = X.selected_text.apply(clean_text)
            text2postag(X, 'cleaned_selected_text', ['pos_', 'dep_', 'text'])
            print('label_sequences...')
            label_sequences(X)
            print('drop_unlabled_sequences...')
            drop_unlabled_sequences(X) # drop samples where (target_sequence == 0).all()
            self.set_of_pos_tags = set(tag for tags in X.cleaned_text_pos_.values for tag in tags if tag != 'SPACE')
            self.pos2idx = dict((tag, index) for index, tag in enumerate(self.set_of_pos_tags))
            self.idx2pos = dict(enumerate(self.set_of_pos_tags))
            self.set_of_dep_tags = set(tag for tags in X.cleaned_text_dep_.values for tag in tags if tag != 'SPACE')
            self.dep2idx = dict((tag, index) for index, tag in enumerate(self.set_of_dep_tags))
            self.idx2dep = dict(enumerate(self.set_of_dep_tags))
        print('pos and dep encoding...')
        X['pos_tag_encoded'] = encode_all(X.cleaned_text_pos_.values, self.set_of_pos_tags, self.pos2idx, self.cat_encoding_func)
        X['dep_tag_encoded'] = encode_all(X.cleaned_text_dep_.values, self.set_of_dep_tags, self.dep2idx, self.cat_encoding_func)
        print('calculate_texts_polarity_score...')
        calculate_texts_polarity_score(X, sentiment_analyzers=self.sentiment_analyzers)
    
        # calculate number of sentences
        X['cleaned_text_num_sents'] = X.cleaned_text.apply(count_sentences)
        # calculate number of tokens (words + punctuation)
        X['cleaned_text_num_tokens'] = X.cleaned_text_text.apply(len)

        # min_max scaling
        if self.isTrain:
            self.max_num_sents = X.cleaned_text_num_sents.max()
            self.max_num_tokens = X.cleaned_text_num_tokens.max()
        X['cleaned_text_num_sents'] = X.cleaned_text_num_sents.values / self.max_num_sents
        X['cleaned_text_num_tokens'] = X.cleaned_text_num_tokens.values / self.max_num_tokens
        
        # concatenate features
        print('concatenating...')
        if self.isTrain:
            self.concatenate_feature_columns = X.columns[X.columns.tolist().index('pos_tag_encoded'):]
        features = concat_encoded_features(X, self.concatenate_feature_columns)
        if self.isTrain:
            target = X.target_sequence.values
            target_weights = X.target_sequence_weights.values
            return X, features, target, target_weights
        return X, features
        
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


preprocessor = Preprocessor(onehot_encode)
new_train, features, target, target_weights = preprocessor.fit_transform(train)
MAX_SEQ_LENGTH = max(len(seq) for seq in features)
print('MAX_SEQ_LENGTH:', MAX_SEQ_LENGTH)

features, target, target_weights = pad_sequences(features, target, target_weights, MAX_SEQ_LENGTH, mode='uniform')
features = np.array(features)
target = np.array(target)
target_weights = np.array(target_weights)
target = target.reshape(*target.shape[:2], 1)
target_weights = target_weights.reshape(*target_weights.shape[:2], 1)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

train_index, val_index, _, _ = train_test_split(new_train.index, new_train.sentiment.values, test_size=.15, random_state=123, shuffle=True, stratify=new_train.sentiment.values)
train_sequences, val_sequences = features[train_index], features[val_index]
train_target, val_target = target[train_index], target[val_index]
train_target_weights, val_target_weights = target_weights[train_index], target_weights[val_index]
train_sequences.shape, val_sequences.shape, train_target.shape, train_target_weights.shape
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Bidirectional, LSTM, Dense, Dropout, GRU, Activation
# from keras_contrib.layers import CRF
model = Sequential()
model.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.2, return_sequences=True), input_shape=features.shape[-2:]))
model.add(Bidirectional(LSTM(units=64, recurrent_dropout=0.2, return_sequences=True)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.003), 
              loss=tf.keras.losses.binary_crossentropy, 
              sample_weight_mode='temporal', 
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name='auc', curve='PR'),
                      tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
model.summary()
model.fit(train_sequences, train_target, 
          batch_size=32, epochs=10, 
          validation_data=(val_sequences, val_target, val_target_weights.reshape(*val_target_weights.shape[:2])), 
          sample_weight=train_target_weights.reshape(*train_target_weights.shape[:2]),
         verbose=2)
pseudo_features = np.array(concat_encoded_features(new_train, preprocessor.concatenate_feature_columns))
pseudo_features.shape
def extract_words(model, features, train, index=None):
    if index is None:
        index = train.index.values
    words = [0] * len(features)
    num_tokens = train.loc[index, 'cleaned_text_num_tokens'].unique()
    for num_tokens_ in num_tokens:
        index_ = train[train.loc[index, 'cleaned_text_num_tokens'] == num_tokens_].index.values
        features_ = np.concatenate(features[index_]).reshape(len(index_), -1, features[0].shape[-1])
        pred = model(features_).numpy().reshape(len(index_), -1)
        wrds = [[word for index2, word in enumerate(words_) if pred[index1][index2] >= 0.5] for index1, words_ in enumerate(train.loc[index_, 'cleaned_text_text'].values)]
        for index1 in range(len(index_)):
            words[index_[index1]] = wrds[index1]
            
    return np.array(words)[index]
        
        
        
tm = time.time()
predicted_words = extract_words(model, pseudo_features, new_train)
print(time.time() - tm)
def jaccard(str1, str2): 
    a = {word.lower() for word in str1}# if word not in string.punctuation}
    b = {word.lower() for word in str2}# if word not in string.punctuation}
    if not a and not b:
        return -1
#     print(a, b, f'"{" ".join(str1)}"', f'"{" ".join(str2)}"')
    c = a.intersection(b)
    return len(c) / (len(a) + len(b) - len(c))

def evaluate(y_true, y_pred):
    score = []
    for i in range(len(y_true)):
        jac_score = jaccard(y_true[i], y_pred[i])
        if jac_score == -1:
            continue
        score.append(jac_score)
    return sum(score) / len(score)
print('all dataset jacard_score: {}, train: {}, val: {}'.format(evaluate(predicted_words, new_train['cleaned_selected_text_text'].values), 
                                                                evaluate(predicted_words[train_index], new_train.loc[train_index, 'cleaned_selected_text_text'].values), 
                                                                evaluate(predicted_words[val_index], new_train.loc[val_index, 'cleaned_selected_text_text'].values)
                                                               )
     )
preprocessor.isTrain = False
new_test, test_features = preprocessor.transform(test)
import transformers
import tensorflow as tf
MODELNAME = 'bert-base-uncased' #distilbert-base-uncased
tokenizer = transformers.AutoTokenizer.from_pretrained(MODELNAME)
BERT = transformers.TFAutoModel.from_pretrained(MODELNAME)
# tokenizer.tokenize(train.loc[0].text), tokenizer.tokenize('somethere?!?!')
MODELNAME = 'albert-base-v2' # '▁' E.g.: ['▁spent', '▁the', '▁entire',]
tokenizer = transformers.AutoTokenizer.from_pretrained(MODELNAME)
BERT = transformers.TFAutoModel.from_pretrained(MODELNAME)
# tokenizer.tokenize(train.loc[0].text), tokenizer.tokenize('somethere?!?!')
MODELNAME = 'roberta-base' #distilroberta-base # 'Ġ' E.g.: ['Sp', 'ent', 'Ġthe','Ġentire',]
tokenizer = transformers.AutoTokenizer.from_pretrained(MODELNAME)
BERT = transformers.TFAutoModel.from_pretrained(MODELNAME)
# tokenizer.tokenize(train.loc[0].text.lower()), tokenizer.tokenize('somethere')
res = BERT(np.array([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train.loc[0].text))]))
res[0].shape, res[1].shape
tokenizer.tokenize('nothing...i think'), tokenizer.tokenize('pre?!?!else somethere predefined')
tokenizer.convert_tokens_to_string(tokenizer.tokenize('nothing?!?!else somethere'))
MODELNAME
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
#     text = re.sub('&quot;', '"', text)
#     text = re.sub('&amp;', '&', text)
    # replace multiple non-latin and non-digit chars into one char. E.g.: '!!!!' into '!'
#     text = re.sub(r'([^a-zA-Z0-9\s])\1+', r'\g<1>', text) 
    # e.g.: replace 'nothing.to.do' into 'nothing. to. do' 
    # in order to interpret them as different tokens for vader.sentimentAnalyzer
#     text = re.sub(r'([?!.])', '\g<1> ', text) 
    return text.lower().strip()

def bert_tokenize(train, column, tokenizer):
    bert_tokens = []
    for text, sentiment in train[[column, 'sentiment']].values:
#         if 'selected' not in column:
        bert_tokens.append([101, sentiment_map[sentiment], 102] + tokenizer.tokenize(text) + [102])
#         else:
#             bert_tokens.append(tokenizer.tokenize(text))
    
    if 'selected' in column:
        train['selected_bert_tokens'] = bert_tokens
    else:
        train['bert_tokens'] = bert_tokens
        

def bertTokens2text(train, column):
    def bertTokens2words_local(tokens): # bert and distilbert
        '''
        Convert bert tokens kinda ['some', '##ther', '##e'] into ['somethere', ...] and [number of byte-pairs (3 in that case), ...] for pos_tagging
        '''
        words = []
        words_bpe_length = []
        for index, token in enumerate(tokens[3:-1]):
            if token[0] == '#' and token[-1] != '#':
                words[-1] += token.replace('#', '')
                words_bpe_length[-1] += 1
            else:
                words.append(token)
                words_bpe_length.append(1)
        return ' '.join(words).strip(), words_bpe_length
    def albertTokens2words_local(tokens): # albert
        '''
        Convert bert tokens kinda ['▁some', 'there'] into ['somethere', ...] and [number of byte-pairs (2 in that case), ...] for pos_tagging
        '''
        words = []
        words_bpe_length = []
        for index, token in enumerate(tokens[3:-1]):
            if token[0] != '▁' and token not in string.punctuation:# and (len(words[-1])>1 or words[-1] not in string.punctuation): 
                words[-1] += token
                words_bpe_length[-1] += 1
            elif token == '▁':
                words.append(token)
                words_bpe_length.append(1)
            else:
                words.append(token.replace('▁', ''))
                words_bpe_length.append(1)
        return ' '.join(words).strip(), words_bpe_length
    def robertTokens2words_local(tokens): # robert / distilrobert
        '''
        Convert bert tokens kinda ['s', 'ome', 'here'] into ['somethere', ...] and [number of byte-pairs (3 in that case), ...] for pos_tagging
        '''
        words = []
        words_bpe_length = []
        for index, token in enumerate(tokens[3:-1]):
            if words and token[0] != 'Ġ' and len(token)>1 and token not in string.punctuation:
                words[-1] += token
                words_bpe_length[-1] += 1
            elif token == 'Ġ':
                words.append(token)
                words_bpe_length.append(1)
            else:
                words.append(token.replace('Ġ', ''))
                words_bpe_length.append(1)
        return ' '.join(words).strip(), words_bpe_length
    
    bertTokens_texts = []
    bertTokens_words_bpe_length = []
    for bertTokens in train[column].values:
        if 'albert' in MODELNAME: # albert
            text, words_bpe_length = albertTokens2words_local(bertTokens)
        elif 'robert' in MODELNAME: # robert or distilrobert
            text, words_bpe_length = robertTokens2words_local(bertTokens)
        else: # bert or distilbert
            text, words_bpe_length = bertTokens2words_local(bertTokens)
        bertTokens_texts.append(text)
        bertTokens_words_bpe_length.append(words_bpe_length)
    train[f'{column}_texts'] = bertTokens_texts
    if 'selected' not in column:
        train[f'{column}_words_bpe_length'] = bertTokens_words_bpe_length
        
# def bert_label_sequences(train):
#     target_sequences = []
#     for index, sequence in enumerate(train.bert_tokens_texts.values):
#         sequence = sequence.split()
#         bert_tokens_words_bpe_length = train.loc[index, 'bert_tokens_words_bpe_length']
#         selected_sequence = train.loc[index, 'selected_bert_tokens_texts'].split()
#         labeled_sequence = np.zeros(len(train.loc[index, 'bert_tokens']), dtype=np.int)
#         if not selected_sequence:
#             target_sequences.append(labeled_sequence)
#             continue
#         local_index = 0
#         labeled_sequence_index = 0
#         while len(sequence) >= local_index + len(selected_sequence):
#             if sequence[local_index : local_index+len(selected_sequence)] == selected_sequence:
#                 for jndex, val in enumerate(bert_tokens_words_bpe_length[local_index:]):
#                     if local_index + jndex >= local_index + len(selected_sequence):
#                         break
#                     labeled_sequence[labeled_sequence_index : labeled_sequence_index + val] = 1
#                     labeled_sequence_index += val
                    
#                 break
#             labeled_sequence_index += bert_tokens_words_bpe_length[local_index]
#             local_index += 1
#         else:
#             local_index = 0
#             labeled_sequence_index = 0
#             while len(sequence) >= local_index + len(selected_sequence):
#                 if (sequence[local_index][-len(selected_sequence[0]):] == selected_sequence[0] and 
#                     sequence[local_index + len(selected_sequence) - 1][:len(selected_sequence[-1])] == selected_sequence[-1] and 
#                     sequence[local_index+1 : local_index+len(selected_sequence)-1] == selected_sequence[1:-1]):
                        
#                     reserved = labeled_sequence_index
                    
#                     for jndex, val in enumerate(bert_tokens_words_bpe_length[local_index:]):
#                         if local_index + jndex >= local_index + len(selected_sequence):
#                             break
#                         labeled_sequence[labeled_sequence_index : labeled_sequence_index + val] = 1
#                         labeled_sequence_index += val
                        
#                     if sequence[local_index] != selected_sequence[0]:
#                         labeled_sequence[reserved : reserved + bert_tokens_words_bpe_length[local_index]] = 0

#                     if sequence[local_index + len(selected_sequence) - 1] != selected_sequence[-1]:
#                         labeled_sequence_index -= bert_tokens_words_bpe_length[local_index + len(selected_sequence) - 1]
#                         labeled_sequence[labeled_sequence_index : labeled_sequence_index + bert_tokens_words_bpe_length[local_index + len(selected_sequence) - 1]] = 0
#                     break
#                 labeled_sequence_index += bert_tokens_words_bpe_length[local_index]
#                 local_index += 1
#         target_sequences.append(labeled_sequence)
#     train['target_sequence'] = target_sequences

def bert_label_sequences(train):
    target_sequences = []
    for index, sequence in enumerate(train.bert_tokens_texts.values):
        sequence = sequence.split()
        bert_tokens_words_bpe_length = train.loc[index, 'bert_tokens_words_bpe_length']
        selected_sequence = train.loc[index, 'selected_bert_tokens_texts'].split()
        labeled_sequence = [0, 0]
        if not selected_sequence:
            target_sequences.append([0, 0])
            continue
        local_index = 0
        labeled_sequence_index = 0
        while len(sequence) >= local_index + len(selected_sequence):
            if sequence[local_index : local_index+len(selected_sequence)] == selected_sequence:
                local_index += 3
                labeled_sequence = [local_index, local_index + len(selected_sequence)]
                    
                break
            labeled_sequence_index += bert_tokens_words_bpe_length[local_index]
            local_index += 1
        else:
            local_index = 0
            labeled_sequence_index = 0
            while len(sequence) >= local_index + len(selected_sequence):
                if (sequence[local_index][-len(selected_sequence[0]):] == selected_sequence[0] and 
                    sequence[local_index + len(selected_sequence) - 1][:len(selected_sequence[-1])] == selected_sequence[-1] and 
                    sequence[local_index+1 : local_index+len(selected_sequence)-1] == selected_sequence[1:-1]):
                        
                    labeled_sequence = [local_index + 3, local_index + len(selected_sequence) + 3]
                        
                    if sequence[local_index] != selected_sequence[0]:
                        labeled_sequence[0] += 1

                    if sequence[local_index + len(selected_sequence) - 1] != selected_sequence[-1]:
                        labeled_sequence[1] -= 1
                    break
                labeled_sequence_index += bert_tokens_words_bpe_length[local_index]
                local_index += 1
        if labeled_sequence[1] <= labeled_sequence[0]:
            labeled_sequence = [0, 0]
        target_sequences.append(labeled_sequence)
    train['target_sequence'] = target_sequences   
        
def drop_unlabled_sequences(train):
    cnt = 0
    drop_indexes = []
    for index, seq in enumerate(train.target_sequence.values):
        if not any(seq):
            cnt += 1
            drop_indexes += [index]
    if drop_indexes:
        train.drop(drop_indexes, inplace=True, axis=0)
        train.reset_index(drop=True, inplace=True)
    print(f'dropped {cnt} samples')
    
def text2tag(train, column, attributes):
    attr_values = [[] for _ in attributes]
    for text in train[column].values:
        for index, attr in enumerate(attributes):
            attr_values[index].append([])
        for token in nlp(text):
            if token.pos_ == 'SPACE':
                continue
            for index, attr in enumerate(attributes):
                attr_values[index][-1].append(getattr(token, attr))
    for index, attr in enumerate(attributes):
        train[f'{"_".join(column.split("_")[:-1])}_{attr}'] = attr_values[index]

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

def calculate_texts_polarity_score(train, n_gram_window_range=(0, 4), sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer()), 
                                                                                            SentimentAnalyzer(TextBlob('123'))]):
    
    column_basename = 'words_polarity_scores'
    polarity_scores = []
    for _ in range(len(sentiment_analyzers)):
        polarity_scores.append([])
        
    for words, sentiment in train[['bert_tokens_texts', 'sentiment']].values:
        words = words.split()
        scores = words_polarity_score(words, sentiment, n_gram_window_range, sentiment_analyzers)
        for index, scores_ in enumerate(scores):
            polarity_scores[index].append(scores_)
            
    for index, sentiment_analyzer in enumerate(sentiment_analyzers):
        if isinstance(sentiment_analyzer.sentiment_analyzer, SentimentIntensityAnalyzer):
            train[f'vader_{column_basename}'] = polarity_scores[index]
        elif isinstance(sentiment_analyzer.sentiment_analyzer, TextBlob):
            train[f'textblob_{column_basename}'] = polarity_scores[index]
#         elif # add your own sentiment analyzer
    return

def count_sentences(text):
    return len(re.split('[.!?]+', text))

def concat_encoded_features(train, columns):
    concateneted_features = []
    for row in train[columns].values:
        concateneted_features.append(np.concatenate(row, axis=1))
    return concateneted_features





class Preprocessor():

    def __init__(self, cat_encoding_func, sentiment_analyzers=[SentimentAnalyzer(SentimentIntensityAnalyzer())], tokenizer=tokenizer):
        self.isTrain = True
        self.cat_encoding_func = cat_encoding_func
        self.sentiment_analyzers = sentiment_analyzers
        self.tokenizer = tokenizer
    
    def fit(self, X, y=None):
        self.isTrain = True
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.isTrain:
            X.fillna('', inplace=True)
        # drop neutral samples, because we would select the whole text for neutral sentiment text
        X.drop(X[X.sentiment == 'neutral'].index, inplace=True)
        X.reset_index(drop=True, inplace=True)
        # clean texts
        print('Clean text...')
        X['cleaned_text'] = X.text.apply(clean_text)
        if self.isTrain:
            X['cleaned_selected_text'] = X.selected_text.apply(clean_text)
        for column in ['cleaned_text', 'cleaned_selected_text']:
            if 'selected' in column:
                if not self.isTrain:
                    continue
                bert_tokenize(X, column, self.tokenizer)
                bertTokens2text(X, 'selected_' + 'bert_tokens') # column = 'bert_tokens' either 'bert_selected_tokens'
            else:
                bert_tokenize(X, column, self.tokenizer)
                bertTokens2text(X, 'bert_tokens') # column = 'bert_tokens' either 'bert_selected_tokens'
        print('text2postag...')
        text2tag(X, 'bert_tokens_texts', ['pos_', 'dep_'])
        if self.isTrain:
            print('label_sequences...')
            bert_label_sequences(X)
            print('drop_unlabled_sequences...')
            drop_unlabled_sequences(X) # drop samples where (target_sequence == 0).all()
            self.set_of_pos_tags = set(tag for tags in X.bert_tokens_pos_.values for tag in tags[3:-1] if tag != 'SPACE')
            self.pos2idx = dict((tag, index) for index, tag in enumerate(self.set_of_pos_tags))
            self.idx2pos = dict(enumerate(self.set_of_pos_tags))
            self.set_of_dep_tags = set(tag for tags in X.bert_tokens_dep_.values for tag in tags[3:-1] if tag != 'SPACE')
            self.dep2idx = dict((tag, index) for index, tag in enumerate(self.set_of_dep_tags))
            self.idx2dep = dict(enumerate(self.set_of_dep_tags))
            
        print('calculate_texts_polarity_score...')
        calculate_texts_polarity_score(X, sentiment_analyzers=self.sentiment_analyzers)
    
        # calculate number of sentences
        X['cleaned_text_num_sents'] = X.cleaned_text.apply(count_sentences)
        # calculate number of tokens (words + punctuation)
        X['cleaned_text_num_tokens'] = X.bert_tokens_texts.apply(lambda x: len(x.split()))

        # min_max scaling
        if self.isTrain:
            self.max_num_sents = X.cleaned_text_num_sents.max()
            self.max_num_tokens = X.cleaned_text_num_tokens.max()
        X['cleaned_text_num_sents'] = X.cleaned_text_num_sents.values / self.max_num_sents
        X['cleaned_text_num_tokens'] = X.cleaned_text_num_tokens.values / self.max_num_tokens
        
        # expand features for bert tokens
        tag_bert_tokens(X, ['bert_tokens_pos_', 'bert_tokens_dep_', 'cleaned_text_num_sents', 'cleaned_text_num_tokens', 'vader_words_polarity_scores'])
        
        print('pos and dep encoding...')
        X['bert_pos_tag_encoded'] = encode_all(X.bert_tokens_pos_.values, self.set_of_pos_tags, self.pos2idx, self.cat_encoding_func)
        X['bert_dep_tag_encoded'] = encode_all(X.bert_tokens_dep_.values, self.set_of_dep_tags, self.dep2idx, self.cat_encoding_func)
#         print(X.head())
        # concatenate features
        print('concatenating...')
        if self.isTrain:
            self.concatenate_feature_columns = ['bert_pos_tag_encoded', 'bert_dep_tag_encoded', 'cleaned_text_num_sents', 'cleaned_text_num_tokens', 'vader_words_polarity_scores']
#         for col in self.concatenate_feature_columns:
#             print(np.array(X.loc[0, col]).shape)
        features = concat_encoded_features(X, self.concatenate_feature_columns)
        if self.isTrain:
            target = X.target_sequence.values
            return X, features, target
        return X, features
        
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

preprocessor = Preprocessor(onehot_encode)
new_train, features, target = preprocessor.fit_transform(train)
preprocessor.isTrain = False
new_test, test_features = preprocessor.transform(test)
MAX_LEN = 128
# for bert_tokens in new_train[new_train.sentiment != 'neutral'].bert_tokens.values:
#     MAX_LEN = max(MAX_LEN, len(bert_tokens))
# print(f'MAX_LEN: {MAX_LEN}')
encoded_text = []
attention_masks = []
for bert_tokens, sentiment in new_train[new_train.sentiment != 'neutral'][['bert_tokens', 'sentiment']].values:
    encoded_text.append([101, sentiment_map[sentiment], 102] + tokenizer.encode(bert_tokens[3:-1], add_special_tokens=False) + [102] + [0]*(MAX_LEN - len(bert_tokens)))
    attention_masks.append([1]*len(bert_tokens) + [0]*(MAX_LEN - len(bert_tokens)))
encoded_text = np.array(encoded_text)
attention_masks = np.array(attention_masks)
encoded_text[:10].shape
# def pad_sequence(feature_sequence, target_sequence, max_seq_len, mode='uniform'):
#     """
#     mode: uniform - num of zero-padding for pre and post padding is aprxmtly the same
#           pre - ...
#           post - ...
#     """
#     if mode == 'uniform':
#         pre_pad_length = (max_seq_len - len(feature_sequence)) // 2
#         pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
#         post_pad_length = max_seq_len - len(feature_sequence) - pre_pad_length
#         post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
#         feature_sequence = (feature_sequence,)
#         target_sequence = (target_sequence,)
#         if pre_pad:
#             feature_sequence = (pre_pad,) + feature_sequence
#             target_sequence = ([0]*pre_pad_length,) + target_sequence
#         if post_pad:
#             feature_sequence += (post_pad,)
#             target_sequence += ([0]*post_pad_length,)
#         if len(feature_sequence) > 1:
#             feature_sequence = np.concatenate(feature_sequence, axis=0)
#             target_sequence = np.concatenate(target_sequence, axis=0)
#         else:
#             feature_sequence, target_sequence = feature_sequence[0], target_sequence[0]
#     elif mode == 'pre':
#         pre_pad_length = max_seq_len - len(feature_sequence)
#         pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
#         feature_sequence = (feature_sequence,)
#         target_sequence = (target_sequence,)
#         if pre_pad:
#             feature_sequence = (pre_pad,) + feature_sequence
#             target_sequence = ([0]*pre_pad_length,) + target_sequence
#         if len(feature_sequence) > 1:
#             feature_sequence = np.concatenate(feature_sequence, axis=0)
#             target_sequence = np.concatenate(target_sequence, axis=0)
#         else:
#             feature_sequence, target_sequence = feature_sequence[0], target_sequence[0]
#     elif mode == 'post':
#         post_pad_length = max_seq_len - len(feature_sequence)
#         post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
#         feature_sequence = (feature_sequence,)
#         target_sequence = (target_sequence,)
#         if post_pad:
#             feature_sequence += (post_pad,)
#             target_sequence += ([0]*post_pad_length,)
#         if len(feature_sequence) > 1:
#             feature_sequence = np.concatenate(feature_sequence, axis=0)
#             target_sequence = np.concatenate(target_sequence, axis=0)
#         else:
#             feature_sequence, target_sequence = feature_sequence[0], target_sequence[0]
#     else:
#         raise BaseException(f'Unknown mode: {mode}. Choose one of the following: ["uniform","pre", "post"]')
#     return feature_sequence, target_sequence

# def pad_sequences(features, target, max_seq_len, mode='uniform'):
#     features, target = list(zip(*[pad_sequence(features[index], target[index], max_seq_len, mode) \
#                                                for index in range(len(features))]))
#     return features, target

# padded_features, padded_target = pad_sequences(features, target, MAX_LEN, mode='post')
# padded_features = np.array(padded_features)
# padded_target = np.array(padded_target)
# padded_target = padded_target.reshape(*padded_target.shape[:2], 1)

def pad_sequence(feature_sequence, max_seq_len, mode='uniform'):
    """
    mode: uniform - num of zero-padding for pre and post padding is aprxmtly the same
          pre - ...
          post - ...
    """
    if mode == 'uniform':
        pre_pad_length = (max_seq_len - len(feature_sequence)) // 2
        pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
        post_pad_length = max_seq_len - len(feature_sequence) - pre_pad_length
        post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
        feature_sequence = (feature_sequence,)
        if pre_pad:
            feature_sequence = (pre_pad,) + feature_sequence
        if post_pad:
            feature_sequence += (post_pad,)
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
        else:
            feature_sequence = feature_sequence[0]
    elif mode == 'pre':
        pre_pad_length = max_seq_len - len(feature_sequence)
        pre_pad = [[0]*len(feature_sequence[0])]*pre_pad_length
        feature_sequence = (feature_sequence,)
        if pre_pad:
            feature_sequence = (pre_pad,) + feature_sequence
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
        else:
            feature_sequence = feature_sequence[0]
    elif mode == 'post':
        post_pad_length = max_seq_len - len(feature_sequence)
        post_pad = [[0]*len(feature_sequence[0])]*post_pad_length
        feature_sequence = (feature_sequence,)
        if post_pad:
            feature_sequence += (post_pad,)
        if len(feature_sequence) > 1:
            feature_sequence = np.concatenate(feature_sequence, axis=0)
        else:
            feature_sequence = feature_sequence[0]
    else:
        raise BaseException(f'Unknown mode: {mode}. Choose one of the following: ["uniform","pre", "post"]')
    return feature_sequence

def pad_sequences(features, max_seq_len, mode='uniform'):
    features = [pad_sequence(features[index], max_seq_len, mode) for index in range(len(features))]
    return features

padded_features = pad_sequences(features, MAX_LEN, mode='post')
padded_features = np.array(padded_features)

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

test_size = .15
random_state = 123
shuffle = True
stratify = new_train.sentiment.values
train_index, val_index, _, _ = train_test_split(new_train.index, new_train.sentiment.values, 
                                                test_size=test_size, random_state=random_state, 
                                                shuffle=shuffle, stratify=stratify)
train_sequences = [encoded_text[train_index], attention_masks[train_index], padded_features[train_index]]
val_sequences = [encoded_text[val_index], attention_masks[val_index], padded_features[val_index]]
train_target, val_target = target[train_index], target[val_index]
train_target = np.concatenate(train_target).reshape(-1, 2)
val_target = np.concatenate(val_target).reshape(-1, 2)
# train_target, val_target = padded_target[train_index], padded_target[val_index]


from tensorflow.keras import Sequential, Model, Input
import tensorflow.keras.layers as L
from tensorflow.keras.layers import InputLayer, Bidirectional, LSTM, Dense, Dropout, GRU, Activation, Concatenate
from tensorflow.keras.initializers import TruncatedNormal

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = BERT
        self.dropout = L.Dropout(0.1)
        self.qa_outputs = L.Dense(2, 
#                                 kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                                dtype='float32',
                                name="qa_outputs")
        self.aux_input = InputLayer(input_shape=(None, 68), )
        self.concat = Concatenate(axis=-1)
#         self.birnn = Bidirectional(LSTM(units=128, recurrent_dropout=0.2, return_sequences=True))#, input_shape=features.shape[-2:])
#         self.fc = Dense(64, activation='relu')
#         self.dropout = Dropout(0.2)
#         self.out = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        last_out = self.bert(inputs[0], attention_mask=inputs[1])[0]
        
#         hidden_states = self.concat([
#             hidden_states[-i] for i in range(1, self.NUM_HIDDEN_STATES+1)
#         ])
#         hidden_states = self.birnn(hidden_states, training=kwargs.get("training", False))
#         hidden_states = self.dropout(hidden_states, training=kwargs.get("training", False))
#         aux_inp = self.aux_input()
#         last_out = self.concat([last_out, inputs[2]])
        hidden_states = self.dropout(last_out, training=kwargs.get("training", False))
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        
        return start_logits, end_logits
        
#         bert_out = self.bert(inputs[0], attention_mask=inputs[1])[0]
#         aux_inp = self.aux_input(inputs[2])
#         concat = self.concat([aux_inp, bert_out])
#         x = self.birnn(concat)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = self.out(x)
#         return x



model = MyModel()
# model()

res[0].shape, tf.argmax(tf.nn.softmax(res), axis=-1)
loss_fn =  tf.keras.losses.sparse_categorical_crossentropy
def custom_loss(y_true, y_pred):
#     print(y_true, y_true[:, 0], y_pred)
#     print
    loss  = loss_fn(y_true[:, 0], y_pred[0], from_logits=True)
    loss += loss_fn(y_true[:, 1], y_pred[1], from_logits=True)
    return loss

custom_loss(train_target[:n], res)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn =  tf.keras.losses.sparse_categorical_crossentropy
def custom_loss(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
#     print
    loss  = loss_fn(y_true[0], y_pred[0])#, from_logits=True)
    loss += loss_fn(y_true[1], y_pred[1])#, from_logits=True)
    return loss

model.compile(optimizer=tf.keras.optimizers.Adam(3e-5), 
              loss=custom_loss,
             )
#               sample_weight_mode='temporal')#, 
#               metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name='auc', curve='PR'),
#                       tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, n_epochs, train_n_samples, val_n_samples):
        super(MyCustomCallback, self).__init__()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.train_n_samples = train_n_samples
        self.val_n_samples = val_n_samples

    def on_train_batch_begin(self, batch, logs=None):
        self.local_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_number += 1
        batch_calc_time = time.time() - self.local_time
        epoch_time = batch_calc_time * self.train_n_samples / self.batch_size
        aprxt_time_to_end_epoch = epoch_time - batch_calc_time * self.batch_number
#         print(epoch_time)
#         print(epoch_time, batch_calc_time * self.batch_number, aprxt_time_to_end_epoch)
        print(f'batch_number: {self.batch_number}/{int(self.train_n_samples / self.batch_size) + 1}, batch calculation time: {round(batch_calc_time, 2)} sec, ' +
              f'aprxt_time to end epoch: {round(aprxt_time_to_end_epoch / 60, 2)} min, ' +
              f'aprxmt_time to end training: {round((aprxt_time_to_end_epoch + epoch_time*(self.n_epochs - self.cur_epoch))/ 60, 2)} min', 
#               end="\r", 
              )
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
    
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_number = 0
        self.epoch_start_time = time.time()
        self.cur_epoch = epoch + 1
        print('Current epoch:', self.cur_epoch)
    
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nepoch №:{epoch + 1} has been ended for {round((time.time() - self.epoch_start_time) / 60, 2)} minutes', flush=True)

train_target
batch_size = 32
n_epochs = 9
model.fit(train_sequences, [train_target.T[0], train_target.T[1]], #tf.constant(train_target)
          batch_size=batch_size, epochs=n_epochs, 
          validation_data=(val_sequences, [val_target.T[0], val_target.T[1]]), 
          callbacks=[MyCustomCallback(batch_size, n_epochs, len(train_sequences[0]), len(val_sequences[0]))],
          verbose=2)
tf.Tensor(train_target, dtype=tf.int32)
tf.stack(([1, 2, 3], [3, 4, 5]))
n = 3
res = model([train_sequences[0][:n], train_sequences[1][:n], train_sequences[2][:n]])
res
for ind in range(n):
    print(res[ind][attention_masks[ind].astype(bool)].reshape(-1).shape, res[ind][attention_masks[ind].astype(bool)].reshape(-1))
    predictions = np.round(res[ind][attention_masks[ind].astype(bool)].reshape(-1)).astype(bool)
    print(new_train.loc[ind, 'cleaned_text'], new_train.loc[ind, 'bert_tokens'], predictions, np.array(new_train.loc[ind, 'bert_tokens'])[predictions])
res[attention_masks[:n].astype(bool)].shape
local_string = 'spe nothing?!?!else somethere 321as'
tokens = np.array(tokenizer.tokenize(local_string))
print(tokens)
tokenizer.convert_tokens_to_string(tokens)
res[0].shape, tf.argmax(tf.nn.softmax(res), axis=-1)[:, 0].numpy(), train_target[:n]
def bert_post_processing(source_text, bert_tokens, start_end, pred_start_end):
    # source_text: str 'text' from train/test df
    # predicted_text: str after tokenizer.convert_tokens_to_string method
    source_text = source_text.split(' ')
    
    pred_start_end = (max(pred_start_end[0], 3), min(len(bert_tokens)-1, pred_start_end[1]))
    if pred_start_end[1] < pred_start_end[0]:
        return source_text.split(' ')
    index0 = 0
    predicted_text = []
    local_word = ''
    isInclude = False
    for index, tkn in enumerate(bert_tokens[3:-1]):
        index += 3
        if index > pred_start_end[1]:
            break
        isInclude |= pred_start_end[0] <= index < pred_start_end[1]
#         print(tkn)
        local_word += tkn.replace('##', '')
        if local_word == source_text[index0]:
            if isInclude:
                predicted_text.append(source_text[index0])
            local_word = ''
            index0 += 1
            isInclude = False
#     if not predicted_text and pred_start_end[0] == pred_start_end[1] and 3 <= pred_start_end[0] < len(bert_tokens) - 1:
#         return bert_tokens[pred_start_end[0]]
    return ' '.join(predicted_text)
        

# index = 0

# local_string = new_train.loc[train_index[0], 'cleaned_text']
# bert_tokens = new_train.loc[train_index[0], 'bert_tokens']
# start_end = train_target[index]
# pred_start_end = tf.argmax(tf.nn.softmax(res), axis=-1)[:, 0].numpy()
# print(local_string, bert_tokens, len(bert_tokens), new_train.loc[train_index[0], 'selected_text'], start_end, pred_start_end)
# bert_post_processing(local_string, bert_tokens, start_end, pred_start_end)

def predict_text(model, sequences, indexes, df, pred_start_end=np.array([])):
    if not pred_start_end.shape[0]:
        pred_start_end = tf.argmax(tf.nn.softmax(model.predict(sequences)), axis=-1).numpy().T
#         proba_predictions = model.predict(sequences) # all tokens proba predictions
    text_predictions = [] # selected text predictions
#     print(pred_start_end)
    for index, ind in enumerate(indexes):
        pred_start_end_local = pred_start_end[index]
        true_start_end_local = df.loc[ind, 'target_sequence']
        if pred_start_end_local[0] >= pred_start_end_local[1] or len(df.loc[ind, 'cleaned_text'].split()) <= 3: # select all the text because all values equal to 0 or number of tokens <= 3
            text_predictions.append(df.loc[ind, 'cleaned_text'])
        else: 
            bert_tokens = np.array(df.loc[ind, 'bert_tokens'])
            if 'roberta' in MODELNAME:
                # todo
                if bert_tokens[0][0] != 'Ġ':
                    bert_tokens[0] = 'Ġ' + bert_tokens[0]
                text_predictions.append(roberta_post_processing(bert_tokens, occurences, prediction_mask, mode='expand')) # truncate or expand
#                 text_predictions.append(tokenizer.convert_tokens_to_string(bert_tokens[prediction_mask]).strip()) # truncate or expand
            elif 'albert' in MODELNAME:
                pass
            else: # bert; expand post_processing
#                 print(df.loc[ind, 'cleaned_text'], bert_tokens, len(bert_tokens), df.loc[ind, 'selected_text'], true_start_end_local, pred_start_end_local)
                post_processed_text = bert_post_processing(df.loc[ind, 'cleaned_text'], bert_tokens, true_start_end_local, pred_start_end_local)
                text_predictions.append(post_processed_text)
                
        
    return pred_start_end, text_predictions
    
    
print('train eval...')
train_pred_start_end, train_text_predictions = predict_text(model, train_sequences, train_index, new_train)#, train_pred_start_end)
print('validation eval...')
val_pred_start_end, val_text_predictions = predict_text(model, val_sequences, val_index, new_train)#, val_pred_start_end)
# n = 10
# print('train eval...')
# train_pred_start_end, train_text_predictions = predict_text(model, [train_sequences[0][:n], train_sequences[1][:n], train_sequences[2][:n]], train_index[:n], new_train)
# print('validation eval...')
# val_pred_start_end, val_text_predictions = predict_text(model, [val_sequences[0][:n], val_sequences[1][:n], val_sequences[2][:n]], val_index[:n], new_train)

train_text_predictions[:10], new_train.loc[train_index[:10], 'selected_text'].values.tolist()
def bert_post_processing(source_text, bert_tokens, occurancies, prediction_mask):
    # source_text: str 'text' from train/test df
    # predicted_text: str after tokenizer.convert_tokens_to_string method
    source_text = source_text.split(' ')
    
    index0 = 0
    predicted_text = []
    local_word = ''
    isInclude = False
    for index, tkn in enumerate(bert_tokens):
        isInclude |= prediction_mask[index]
        local_word += tkn.replace('##', '')
        if local_word == source_text[index0]:
            if isInclude:
                predicted_text.append(source_text[index0])
            local_word = ''
            index0 += 1
            isInclude = False
    return ' '.join(predicted_text)
        

# local_string = 'spe nothing?!?!else somethere 321as'
# tokens = np.array(tokenizer.tokenize(local_string))
# mask = np.array([False, False, False, True, True, True, True, False, False, False, False, False, False])
# occ = np.where(mask == True)[0]
# print(mask, tokens, occ, len(tokens), len(mask))
# print(local_string)
# print(bert_post_processing(local_string, tokens, occ, mask))
# local_string = 'nothing?!?!else somethere'
# new_str = local_string[3:]
# tokenizer.convert_tokens_to_string(tokenizer.tokenize(new_str)), post_processing(local_string, tokenizer.convert_tokens_to_string(tokenizer.tokenize(new_str)))
def roberta_post_processing(bert_tokens, occurancies, prediction_mask, mode='expand'):
    if mode == 'expand':
        index = occurancies[0] - 1
        if bert_tokens[occurancies[0]][0] != 'Ġ':
#             print(f'{mode} pre del')
            while index >= 0:
                prediction_mask[index] = True
                if bert_tokens[index][0] == 'Ġ':
                    break
                index -= 1

        index = occurancies[-1] + 1
#         if index != len(bert_tokens) and bert_tokens[index][0] != 'Ġ':
#             print(f'{mode} post del')
        while index != len(bert_tokens) and bert_tokens[index][0] != 'Ġ':
            prediction_mask[index] = True
            index += 1
    elif mode == 'truncate':
        isThereAnyG = [True for _ in bert_tokens[prediction_mask] if _[0] == 'Ġ']
        if isThereAnyG:
            
            if bert_tokens[occurancies[0]][0] != 'Ġ':
                index = 0
#                 print(f'{mode} pre del {len(isThereAnyG)}')
                while index < len(occurancies) and bert_tokens[occurancies[index]][0] != 'Ġ':
                    prediction_mask[occurancies[index]] = False
                    index += 1
                    
            if len(isThereAnyG)>1:
#                 print(f'{mode} post del {len(isThereAnyG)}')
#                 print(bert_tokens[occurancies[-1]][0] != 'Ġ', len(bert_tokens)>occurancies[-1]+1, bert_tokens[occurancies[-1]+1][0] == 'Ġ')
                if len(bert_tokens)>occurancies[-1]+1 and bert_tokens[occurancies[-1]+1][0] != 'Ġ':
                    index = len(occurancies) - 1
#                     print(f'{mode} post del 1 {len(isThereAnyG)}')
                    while index > 0 and bert_tokens[occurancies[index]][0] != 'Ġ':
                        prediction_mask[occurancies[index]] = False
                        index -= 1
                    prediction_mask[occurancies[index]] = False
            else: # len(isThereAnyG) = 1
#                 print(f'{mode} post del {len(isThereAnyG)}')
                index = occurancies[-1] + 1
#                 if index != len(bert_tokens) and bert_tokens[index][0] != 'Ġ':
#                     print(f'{mode} post del')
                while index != len(bert_tokens) and bert_tokens[index][0] != 'Ġ':
                    prediction_mask[index] = True
                    index += 1
        else:
#             print(f'{mode} post del')
            return roberta_post_processing(bert_tokens, occurancies, prediction_mask, mode='expand')
#     print(len(bert_tokens), len(prediction_mask))
#     return prediction_mask, tokenizer.convert_tokens_to_string(bert_tokens[prediction_mask]).strip()
    return tokenizer.convert_tokens_to_string(bert_tokens[prediction_mask]).strip()
    
        
# local_string = ' spe nothing?!?!else somethere 321as'
# tokens = np.array(tokenizer.tokenize(local_string))
# mask = np.array([False, False, True, True, True, True, True, False, False])
# occ = np.where(mask == True)[0]
# print(mask, tokens, occ, len(tokens), len(mask))
# print(roberta_post_processing(tokens, occ, mask, mode='expand'))

def predict_text(model, sequences, indexes, df, proba_predictions=np.array([])):
    if not proba_predictions.shape[0]:
        proba_predictions = model.predict(sequences) # all tokens proba predictions
    text_predictions = [] # selected text predictions
    selected_proba_predictions = []
    for index, ind in enumerate(indexes):
        selected_proba_predictions.append(proba_predictions[index][attention_masks[ind].astype(bool)].reshape(-1))
        prediction_mask = np.round(selected_proba_predictions[-1]).astype(bool)
        occurences = np.where(prediction_mask == True)[0]
        if not occurences.shape[0] or len(df.loc[ind, 'cleaned_text'].split()) <= 3: # select all the text because all values equal to 0 or number of tokens <= 3
            text_predictions.append(df.loc[ind, 'cleaned_text'])
        else: 
            bert_tokens = np.array(df.loc[ind, 'bert_tokens'])
            prediction_mask[occurences[0] : occurences[-1] + 1] = True
            if 'roberta' in MODELNAME:
                if bert_tokens[0][0] != 'Ġ':
                    bert_tokens[0] = 'Ġ' + bert_tokens[0]
                text_predictions.append(roberta_post_processing(bert_tokens, occurences, prediction_mask, mode='expand')) # truncate or expand
#                 text_predictions.append(tokenizer.convert_tokens_to_string(bert_tokens[prediction_mask]).strip()) # truncate or expand
            elif 'albert' in MODELNAME:
                pass
            else: # bert; expand post_processing
                post_processed_text = bert_post_processing(df.loc[ind, 'cleaned_text'], bert_tokens, occurences, prediction_mask)
                text_predictions.append(post_processed_text)
                
        
    return proba_predictions, text_predictions, selected_proba_predictions
    
    
print('train eval...')
train_proba_predictions, train_text_predictions, train_selected_proba_predictions = predict_text(model, train_sequences, train_index, new_train)#, train_proba_predictions)
print('validation eval...')
val_proba_predictions, val_text_predictions, val_selected_proba_predictions = predict_text(model, val_sequences, val_index, new_train)#, val_proba_predictions)
# n = 10
# print('train eval...')
# train_proba_predictions, train_text_predictions, train_selected_proba_predictions = predict_text(model, [train_sequences[0][:n], train_sequences[1][:n], train_sequences[2][:n]], train_index[:n], new_train)
# print('validation eval...')
# val_proba_predictions, val_text_predictions, val_selected_proba_predictions = predict_text(model, [val_sequences[0][:n], val_sequences[1][:n], val_sequences[2][:n]], val_index[:n], new_train)

train_proba_predictions[:2]
train_selected_proba_predictions[:2], train_text_predictions[:2],train_proba_predictions[:2]
tkns = tokenizer.tokenize('thinks tonight couldn\'t have gone more perfect.')
print(tkns)
tkns[0] = 'Ġ' + tkns[0]
tokenizer.convert_tokens_to_string(tkns)
new_train.loc[train_index[0]]
with open('train_proba_predictions.txt', 'w') as f:
#     f.write(str(train_pred) + '\n')
    f.write(str(train_predictions) + '\n')
    f.write(str(train_proba_predictions) + '\n')
#     f.write(str(val_pred) + '\n')
    f.write(str(val_predictions) + '\n')
    f.write(str(val_proba_predictions) + '\n')
print('saved train_val predictions')
print('saving weights...')
model.save_weights('roberta_weights')
train_text_predictions[:5]
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def score_all(indexes, text_predictions):
    jaccard_scores = []
    for ind, index in enumerate(indexes):
        jaccard_scores.append(jaccard(new_train.loc[index, 'cleaned_selected_text'], text_predictions[ind]))
    return jaccard_scores

train_jaccard_scores = score_all(train_index, train_text_predictions)
print('TRAIN JACARD-SCORE:', np.mean(train_jaccard_scores))

val_jaccard_scores = score_all(val_index, val_text_predictions)
print('VAL JACARD-SCORE:', np.mean(val_jaccard_scores))
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def score_all(indexes, text_predictions):
    jaccard_scores = []
    for ind, index in enumerate(indexes):
        jaccard_scores.append(jaccard(new_train.loc[index, 'cleaned_selected_text'], text_predictions[ind]))
    return jaccard_scores

train_jaccard_scores = score_all(train_index, train_text_predictions)
print('TRAIN JACARD-SCORE:', np.mean(train_jaccard_scores))

val_jaccard_scores = score_all(val_index, val_text_predictions)
print('VAL JACARD-SCORE:', np.mean(val_jaccard_scores))
