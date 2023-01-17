import gc

import re

import operator 



import numpy as np

import pandas as pd



from gensim.models import KeyedVectors



from sklearn import model_selection





import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





from sklearn.model_selection import train_test_split, cross_val_score

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,  Callback, EarlyStopping, ReduceLROnPlateau
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

train = train.sample(n=600000)
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train, NAlist = reduce_mem_usage(train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
df = pd.concat([train[['id','comment_text']], test], axis=0)

gc.collect()
IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'
y_train = train[TARGET_COLUMN].values

y_aux_train = train[AUX_COLUMNS].values
print(y_train.shape)

print(y_aux_train.shape)
df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",

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
df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
punct_mapping = {"_":" ", "`":" "}



punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



def clean_special_chars(text, puncts, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])    

    for p in puncts:

        text = text.replace(p, f' {p} ')     

    return text



df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
def clean_numbers(x):

    return re.sub('\d+', ' ', x)

df['comment_text'] = df['comment_text'].apply(clean_numbers)

df['comment_text'] = df['comment_text'].apply(clean_numbers)
gc.collect()



train = df.iloc[:600000,:]

test = df.iloc[600000:,:]



list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]
import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 200

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
inp = Input(shape=(maxlen, ))

embed_size = 128

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
x = Embedding(max_features, embed_size)(inp)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

result = Dense(1, activation='sigmoid')(hidden)

aux_result = Dense(len(AUX_COLUMNS), activation='sigmoid')(hidden)
model = Model(inputs=inp, outputs=[result, aux_result])

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
batch_size = 32

epochs = 2

model.fit(X_t,[y_train, y_aux_train], batch_size=batch_size, epochs=epochs, validation_split=0.1)
predictions = model.predict(X_te)
probabilities = predictions[0]

output_df = pd.DataFrame(probabilities, columns=['prediction'])

merged_df =  pd.concat([test, output_df], axis=1)

submission = merged_df.drop(['comment_text'], axis=1)
submission.to_csv("submission.csv", index=False)