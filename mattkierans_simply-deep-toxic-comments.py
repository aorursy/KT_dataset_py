# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.options.display.max_colwidth = -1
pd.options.display.max_columns = 15
class Data:
    """ Loads and preprocesses data """
    def __init__(self, id_col='id', text_col='comment_text'):
        self.train_df, self.test_df = self.load_data()
        self.text_col = 'comment_text'
        self.id_col = 'id'
        
    def preprocessing(self):
        """ Clean the text in some way """
        return

    def load_data(self):
        train_path = '../input/jigsaw-toxic-comment-classification-challenge/train.csv'
        test_path = '../input/jigsaw-toxic-comment-classification-challenge/test.csv'

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        return train_df, test_df
    
    def get_comments(self, subset='train'):
        if subset == 'train':
            data = list(self.train_df[self.text_col])
        if subset == 'test':
            data = list(self.test_df[self.text_col])
        if subset == 'all':
            data = list(self.train_df[self.text_col]) + list(self.test_df[self.text_col])
        return data
    
    def get_training_labels(self):
        labels_columns = self.train_df.columns.difference([self.text_col, self.id_col])
        labels = self.train_df.loc[:, labels_columns].values
        return labels
# explore data
data = Data()
train_df = data.train_df
train_df.sample(5)
# At Bruce K's last presentation, Matt E asks "What the base rates are for each toxicity" (16:04)
train_df.loc[:, train_df.columns.difference(['id', 'comment_text'])].sum()/len(train_df)
# At Bruce K's last presentation, Matt E asks "If a comment is severely toxic, does that mean that it's toxic as well?" (3:40)
print(train_df.loc[train_df['severe_toxic'] == 1].sample(3))
train_df.loc[train_df['severe_toxic'] == 1, train_df.columns.difference(['id', 'comment_text'])].sum(axis=0)
# Any other questions about the data?
import spacy

from collections import defaultdict


class TextMapper:
    """ Maps text into model input format """
    PADDING_SYMBOL = "<PAD>"
    UNKNOWN_SYMBOL = "<UNK>"
    BASE_ALPHABET = [PADDING_SYMBOL, UNKNOWN_SYMBOL]

    def __init__(self, comment_texts, max_sent_len=400, threshold=20, lowercase=False):
        self.lowercase = lowercase
        self.max_sent_len = max_sent_len

        self.word_to_ix = dict()  # maps words to index values
        self.ix_to_word = dict()  # maps index values to words
        self.corpus_info = dict()  # contains infomation about corpus

        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.calc_corpus_info(comment_texts)
        self.init_mappings(threshold)

    def init_mappings(self, threshold=20, check_coverage=True):
        # what information are we losing about the words?
        word_counts = self.corpus_info['word_counts']
        word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab = [word for word, count in word_counts if count > threshold]
        vocab = self.BASE_ALPHABET + vocab

        # can add more criteria to select/normalize words (ignore punctuation, normalize numbers,.. etc)
        
        self.word_to_ix = {word: ix for ix, word in enumerate(vocab)}
        self.ix_to_word = {ix: word for ix, word in enumerate(vocab)}
        if check_coverage:
            self.print_coverage_statistics()
    
    def calc_corpus_info(self, comment_list):
        self.corpus_info['word_counts'] = defaultdict(int)
        self.corpus_info['sent_lengths'] = []
        self.corpus_info['word_lengths'] = []
        for comment in comment_list:
            if self.lowercase:
                comment = comment.lower()
            tokenized_comment = self.nlp(comment)
            self.corpus_info['sent_lengths'].append(len(tokenized_comment))
            for token in tokenized_comment:
                text = token.text
                self.corpus_info['word_counts'][text] += 1
                self.corpus_info['word_lengths'].append(len(text))
                
    def text_to_x(self, text):
        x = np.zeros(self.max_sent_len)

        if self.lowercase:
            text = text.lower()
        tokenized_comment = self.nlp(text)
        for ind, token in enumerate(tokenized_comment[:self.max_sent_len]):
            word = token.text
            x[ind] = self.get_word_index(word)
        return x

    def get_word_index(self, word):
        try:
            num = self.word_to_ix[word]
        except KeyError:
            num = self.word_to_ix[self.UNKNOWN_SYMBOL]
        return num

    def x_to_text(self, x):
        words = [self.ix_to_word[int(i)] for i in x]
        comment_text = " ".join(words)

        # remove padding
        comment_text = comment_text.split(self.PADDING_SYMBOL)[0]
        return comment_text

    def get_texts_x(self, texts):
        x_rep = np.array([self.text_to_x(text) for text in texts])
        return x_rep
    
    def set_max_sent_length(self, sent_len):
        self.max_sent_len = sent_len
    
    def print_coverage_statistics(self):
        word_mappings = self.word_to_ix.keys()
        print("Number of unique words: {}".format(len(word_mappings)))
        total_tokens = 0
        mapped_tokens = 0
        word_counts = self.corpus_info['word_counts']
        for word, count in word_counts.items():
            total_tokens += count
            if word in word_mappings:
                mapped_tokens += count
        print("Percent of unique words mapped: {}%".format(100*len(word_mappings)/len(word_counts)))
        print("Percent of total tokens mapped: {}%".format(100*mapped_tokens/total_tokens))
# text_mapper = TextMapper(data.get_comments('train'))
# # what are the consequences of using testing data here?
# # tune text mapper word threshold

# text_mapper.init_mappings(threshold=4)
# # other methods of measuring coverage are more accurate but take more time
# # consequence of setting too high / too low thresholds?
# # tune text mapper sentence length
# from scipy import stats

# sentence_lengths = text_mapper.corpus_info['sent_lengths']
# stats.describe(sentence_lengths)
# import matplotlib.pyplot as plt

# plt.hist(sentence_lengths, bins=np.arange(0, 1000, 25), cumulative=True, normed=1)
# plt.hlines(0.975, 0, 1000, colors='red')
# # consequence of setting too high / too low sentence lengths?
# text_mapper.set_max_sent_length(400)
# # initialize pretrained word embeddings

# def load_glove(word_to_ix):
#     EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
#     def get_coefs(word,*arr):
#         return word, np.asarray(arr, dtype='float32')

#     print("Loading embeddings")
#     embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
#     print("Loaded {} embeddings".format(len(embeddings_index)))
    
#     # calculate statistics on distributions in embeddings
#     all_embs = np.stack(embeddings_index.values())
#     emb_mean, emb_std = all_embs.mean(), all_embs.std()
#     embed_size = all_embs.shape[1]
    
#     # question about embeddings: what values are in these vectors?
#     print("Embedding mean: {}\nEmbedding std: {}".format(emb_mean, emb_std))

#     # create embeddings matrix for words in our corpus (word_to_ix)
#     nb_words = len(word_to_ix)
#     embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#     matched_words = 0
#     for word, i in word_to_ix.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#             matched_words += 1
#     print("Percent of words with pretrained embedding {}%".format(100*matched_words/nb_words))
#     return embedding_matrix 
# embedding_matrix = load_glove(text_mapper.word_to_ix)
# embedding_matrix.shape
# full_train_x = text_mapper.get_texts_x(data.get_comments('train'))
# full_train_y = data.get_training_labels()
# # sanity check
# first_n_words = 50

# # randomize for fun
# import random
# i = int(random.random()*10000)
# random_comment = data.get_comments('train')[i]
# print("Random comment \n\n{}\n\n".format(random_comment))
# model_input = text_mapper.get_texts_x([random_comment])[0][:first_n_words]
# print("Model input \n\n{}\n\n".format(model_input))
# model_input_to_text = text_mapper.x_to_text(model_input)
# print("Translated model input \n\n{}\n\n".format(model_input_to_text))
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(full_train_x, full_train_y, test_size=0.1)
# from keras.models import Model
# from keras.layers import Input, Dense, Conv1D, Activation, Embedding, MaxPooling1D, Flatten, Dropout, Bidirectional, GlobalMaxPooling1D, LSTM, SpatialDropout1D
# from keras.layers import CuDNNLSTM, Concatenate, GlobalAveragePooling1D, CuDNNGRU
# sent_len = text_mapper.max_sent_len
# unique_tokens = len(text_mapper.word_to_ix)
# embedding_size = embedding_matrix.shape[1]
# spacial_dropout = 0.5
# lstm_kernel_size = 40
# pred_size = 6

# def simple_model():
#     inp = Input(shape=(sent_len, ), name='word_ixs')
#     embedding = Embedding(input_dim=unique_tokens,
#                           output_dim=embedding_size,
#                           input_length=sent_len,
#                           weights=[embedding_matrix],
#                           name="word_embedding")(inp)

#     spatial_drop_layer = SpatialDropout1D(spacial_dropout)(embedding)
#     lstm = Bidirectional(CuDNNLSTM(lstm_kernel_size, return_sequences=True, name="lstm"))(spatial_drop_layer)
#     gru, h_r, h_l = Bidirectional(CuDNNGRU(lstm_kernel_size, return_sequences=True, name="gru", return_state=True))(lstm)
#     max_pool = GlobalMaxPooling1D(name="global_max_pool")(gru)
#     avg_pool = GlobalAveragePooling1D(name="global_avg_pool")(gru)
#     concat_features = Concatenate()([h_r, max_pool, avg_pool])
#     preds = Dense(pred_size, activation='sigmoid')(concat_features)
#     model = Model(inputs=inp, outputs=preds)
#     return model
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# model_save_path = "simple_model.h5"
# learning_rate = 0.003

# early_stopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1)
# checkpointer = ModelCheckpoint(filepath=model_save_path, monitor='val_acc', save_best_only=True, verbose=1)
# callbacks = [early_stopping, checkpointer]

# simple_model = simple_model()
# simple_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
# simple_model.summary()
# simple_model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), shuffle=True, epochs=1, callbacks=callbacks)
# how can we add more information to the network? Think of all the information the network doesn't know...
# def add_features(df):
#     """ stolen from https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model """
#     df['comment_text'] = df['comment_text'].apply(lambda x:str(x))
#     df['total_length'] = df['comment_text'].apply(len)
#     df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
#     df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
#                                 axis=1)
#     df['num_words'] = df.comment_text.str.count('\S+')
#     df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
#     df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  

#     return df
# train_df = add_features(train_df)
# train_df.sample(10)
# # let's add the feature 'caps_vs_length' to our model
# full_add_feature = np.array([[x] for x in train_df['caps_vs_length'].values])
# full_add_feature[0:2]
# train_text_x, test_text_x, train_add_x, test_add_x, train_labels, test_labels = train_test_split(full_train_x, full_add_feature, full_train_y, test_size=0.1)
# sent_len = text_mapper.max_sent_len
# unique_tokens = len(text_mapper.word_to_ix)
# embedding_size = embedding_matrix.shape[1]
# spacial_dropout = 0.5
# lstm_kernel_size = 40
# pred_size = 6

# def nice_model():
#     inputs = []
#     text_inp = Input(shape=(sent_len, ), name='word_ixs')
#     inputs.append(text_inp)
#     add_inp = Input(shape=(1, ), name='add_feature_inp')
#     inputs.append(add_inp)
#     embedding = Embedding(input_dim=unique_tokens,
#                           output_dim=embedding_size,
#                           input_length=sent_len,
#                           weights=[embedding_matrix],
#                           name="word_embedding")(text_inp)

#     spatial_drop_layer = SpatialDropout1D(spacial_dropout)(embedding)
#     lstm = Bidirectional(CuDNNLSTM(lstm_kernel_size, return_sequences=True, name="lstm"))(spatial_drop_layer)
#     gru, h_r, h_l = Bidirectional(CuDNNGRU(lstm_kernel_size, return_sequences=True, name="gru", return_state=True))(lstm)
#     max_pool = GlobalMaxPooling1D(name="global_max_pool")(gru)
#     avg_pool = GlobalAveragePooling1D(name="global_avg_pool")(gru)
#     concat_features = Concatenate()([h_l, h_r, max_pool, avg_pool, add_inp])
#     preds = Dense(pred_size, activation='sigmoid', name='preds')(concat_features)
#     model = Model(inputs=inputs, outputs=preds)
#     return model
# model_save_path = "nice_model.h5"
# learning_rate = 0.003

# early_stopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1)
# checkpointer = ModelCheckpoint(filepath=model_save_path, monitor='val_acc', save_best_only=True, verbose=1)
# callbacks = [early_stopping, checkpointer]

# nice_model = nice_model()
# nice_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
# nice_model.summary()
# train_x = {}
# train_y = {}
# test_x = {}
# test_y = {}

# train_x['word_ixs'] = train_text_x
# test_x['word_ixs'] = test_text_x
# train_x['add_feature_inp'] = train_add_x
# test_x['add_feature_inp'] = test_add_x

# train_y['preds'] = train_labels
# test_y['preds'] = test_labels
# nice_model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), shuffle=True, epochs=3, callbacks=callbacks)