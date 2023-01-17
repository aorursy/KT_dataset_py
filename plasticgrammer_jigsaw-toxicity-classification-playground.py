import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc,os,sys

import operator 



from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras.optimizers import Adam



sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format
%%time

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



print(train.shape, test.shape)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.head()
test.head()
train['target'].describe()
#train['target'].hist(bins=50, figsize=(10,3))

#sns.distplot(train['target'], bins=50, kde=True)



target_bin = pd.cut(train['target'], [0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1], right=False).value_counts()

target_bin = pd.Series(target_bin)

target_bin.plot.bar(color='navy', figsize=(8,3), title='target histgram (by range)')

target_bin.to_frame().T
train.sort_values(['target'], ascending=False).head()
train.sort_values(['target']).head()
# word-count histgram

word_counts = train['comment_text'].apply(lambda x: len(x.split()))

word_counts.hist(bins=50, figsize=(10,3))



print('max words: ', max(word_counts))

print('sum words: ', sum(word_counts))

del word_counts
print('toxic comment:\n', train[train['target'] == 1]['comment_text'].iloc[0])

print()

print('non-toxic comment:\n', train[train['target'] == 0]['comment_text'].iloc[0])
all_df = pd.concat([train, test], sort=False)

del (train, test)

gc.collect()
all_df['comment_text'].values[0]
all_df['comment_text'] = all_df['comment_text'].apply(lambda x: x.lower())
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "can not", "'cause": "because",

                       "could've": "could have", "couldn't": "could not", "didn't": "did not", 

                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would",

                       "he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                       "I'll've": "I will have","I'm": "I am", "I've": "I have",

                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will",

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not",

                       "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 

                       "it'll've": "it will have","it's": "it is", "let's": "let us",

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                       "mightn't": "might not","mightn't've": "might not have", 

                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                       "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", 

                       "oughtn't've": "ought not have", "shan't": "shall not",

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                       "she'd've": "she would have", "she'll": "she will",

                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                       "shouldn't": "should not", "shouldn't've": "should not have", 

                       "so've": "so have","so's": "so as", "this's": "this is",

                       "that'd": "that would", "that'd've": "that would have", "that's": "that is",

                       "there'd": "there would", "there'd've": "there would have", 

                       "there's": "there is", "here's": "here is","they'd": "they would", 

                       "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", 

                       "they've": "they have", "to've": "to have", "wasn't": "was not",

                       "we'd": "we would", "we'd've": "we would have", "we'll": 

                       "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have",

                       "what're": "what are",  "what's": "what is", "what've": "what have", 

                       "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", 

                       "who'll've": "who will have", "who's": "who is", "who've": "who have", 

                       "why's": "why is", "why've": "why have", "will've": "will have",

                       "won't": "will not", "won't've": "will not have", "would've": "would have",

                       "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have",

                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", 

                       "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



all_df['comment_text'] = all_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
def preprocess(data):

    def clean_special_chars(text):

        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&…'

        for p in punct:

            text = text.replace(p, ' ')

        for p in '0123456789':

            text = text.replace(p, ' ')

        #for p in "?!.,":

        #    text = text.replace(p, ' ' + p)

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x))

    return data



all_df['comment_text'] = preprocess(all_df['comment_text'])
table = str.maketrans('ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘʀᴛᴜᴠᴡʏᴢ', 'abcdefghijklmnoprtuvwyx')

all_df['comment_text'] = all_df['comment_text'].apply(lambda x: x.translate(table))
text_to_word_sequence(all_df['comment_text'].values[0])
train = all_df[all_df['target'].notnull()]

test = all_df[all_df['target'].isnull()]



X_train = train.drop(['id','target'], axis=1)

Y_train = (train['target'] >= 0.5).astype(int)

X_test  = test.drop(['id','target'], axis=1)

#train_id  = train['id']

#test_id  = test['id']



print(X_train.shape, X_test.shape)
del (all_df, train, test)

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:5])
TOXICITY_COLUMN = 'target'

TEXT_COLUMN = 'comment_text'

MAX_NUM_WORDS = 300000

TOKENIZER_FILTER = '\r\t\n'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=TOKENIZER_FILTER)

tokenizer.fit_on_texts(list(X_train[TEXT_COLUMN]) + list(X_test[TEXT_COLUMN]))
counter = sorted(dict(tokenizer.word_docs).items(), key=lambda x:x[1], reverse=True)

wordcount = pd.Series([x[1] for x in counter], [x[0] for x in counter])

del counter



wordcount[:30].plot.bar(color='navy', width=0.7, figsize=(12,3))
tokenizer_tx = Tokenizer(num_words=MAX_NUM_WORDS, filters=TOKENIZER_FILTER)

tokenizer_tx.fit_on_texts(list(X_train.loc[Y_train == 1, TEXT_COLUMN]))



counter = sorted(dict(tokenizer_tx.word_docs).items(), key=lambda x:x[1], reverse=True)

wordcount_tx = pd.Series([x[1] for x in counter], [x[0] for x in counter])



wordcount_stats = pd.concat([wordcount, wordcount_tx], axis=1, keys=[0, 'toxic'], sort=False)

wordcount_only_tx = wordcount_stats[wordcount_stats[0] * 0.8 <= wordcount_stats['toxic']].copy()

wordcount_only_tx.drop('toxic', axis=1, inplace=True)

wordcount_only_tx = wordcount_only_tx[wordcount_only_tx[0] > 1]



print(len(wordcount_only_tx))

wordcount_only_tx[:10]
wordcount = pd.concat([wordcount_only_tx, wordcount])[0]

del counter, wordcount_tx, wordcount_stats, wordcount_only_tx
wordsum = wordcount.sum()



n_words = len(wordcount)

cumsum_rate = wordcount.cumsum() / wordsum

cover_rate = {}

for i in range(100, 90, -1):

    p = i / 100

    cover_rate[str(i)+'%'] = n_words - len(cumsum_rate[cumsum_rate > p])

del cumsum_rate



pd.Series(cover_rate).plot.barh(color='navy', figsize=(12, 3), title='vocab-size by coverage-rate')

pd.Series(cover_rate).to_frame().T
VOCAB_SIZE = 50000



print('covered', wordcount[VOCAB_SIZE], 'times word')
%%time



EMBEDDINGS_DIMENSION = 300

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((VOCAB_SIZE + 1, EMBEDDINGS_DIMENSION))

    unknown_words = []

    for i in range(VOCAB_SIZE):

        try:

            word = wordcount.index[i]

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words



crawl_matrix, unknown_words_crawl = build_matrix(CRAWL_EMBEDDING_PATH)

glove_matrix, unknown_words_glove = build_matrix(GLOVE_EMBEDDING_PATH)



word2index = dict((wordcount.index[i], i) for i in range(VOCAB_SIZE))



embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

#embedding_matrix = glove_matrix

embedding_matrix.shape
words_count = len(unknown_words_crawl)

print('n unknown words (crawl):', words_count, ', {:.3%} of all words'.format(words_count / n_words))

print('unknown words (crawl):', unknown_words_crawl)

words_count = len(unknown_words_glove)

print('n unknown words (glove):', words_count, ', {:.3%} of all words'.format(words_count / n_words))

print('unknown words (glove):', unknown_words_glove)
del crawl_matrix, unknown_words_crawl

del glove_matrix, unknown_words_glove

del wordcount

gc.collect()
MAX_SEQUENCE_LENGTH = 256



def word_index(word):

    try:

        return word2index[word]

    except KeyError:

        return VOCAB_SIZE



# All comments must be truncated or padded to be the same length.

def pad_text(texts, tokenizer):

    matrix = [list(map(word_index, text_to_word_sequence(t, filters=TOKENIZER_FILTER))) for t in texts]

    return pad_sequences(matrix, maxlen=MAX_SEQUENCE_LENGTH)



train_text = pad_text(X_train[TEXT_COLUMN], tokenizer)

test_text = pad_text(X_test[TEXT_COLUMN], tokenizer)
del (X_train, X_test)

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
def build_model(lr=0.0, lr_d=0.0, units=64, spatial_dr=0.0, 

                dense_units=0, dr=0.1, conv_size=32, epochs=20):

    

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)



    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(*embedding_matrix.shape,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

    

    x = embedding_layer(sequence_input)

    x = SpatialDropout1D(spatial_dr)(x)

    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)   

    x = Conv1D(conv_size, 2, padding="valid", kernel_initializer="he_uniform")(x)

  

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)     

    

    x = concatenate([avg_pool1, max_pool1])

    x = BatchNormalization()(x)

    x = Dense(int(dense_units / 2), activation='relu')(x)

    x = Dropout(dr)(x)

    

    preds = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

    model.summary()

    history = model.fit(train_text, Y_train, batch_size=1024, epochs=epochs, validation_split=0.1, 

                        verbose=1, callbacks=[check_point, early_stop])

   

    model = load_model(file_path)

    return model
model = build_model(lr=1e-3, lr_d=1e-7, units=64, spatial_dr=0.2, dense_units=64, dr=0, conv_size=64, epochs=20)

pred = model.predict(test_text)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = pred

submission.reset_index(drop=False, inplace=True)

submission.to_csv('submission.csv', index=False)

submission.head()
submission['prediction'].describe()
target_bin = pd.cut(submission['prediction'], [0, 0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1], right=False).value_counts()

target_bin = pd.Series(target_bin)

target_bin.plot.bar(color='navy', figsize=(10,3))

target_bin.to_frame().T