!pip install gensim --upgrade
!pip install keras --upgrade
!pip install pandas --upgrade
# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
%matplotlib inline

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')

# DATASET
"""
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8
"""

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"
raw_data = pd.read_csv("../input/hmif-challenge-2019/train.csv")
test_data = pd.read_csv("../input/hmif-challenge-2019/test.csv")

raw_data.head()                    
print(len(raw_data)) 
print(len(test_data))
raw_data.dtypes

for col in raw_data.columns:
    
    if (str(col) != "captions") and (raw_data[col].dtypes != raw_data["post_id"].dtypes) :
        print("------------------")
        print("\n " + str(col) + " : \n")
        print( raw_data[col].value_counts())
        print("\n\n------------------------")



caption_string_vec = []

# Subtitute emoji

import re


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


for caption in raw_data["captions"]:
    
    caption_string_vec.append(str(deEmojify(caption)))
    
#caption_string_vec

# Detect language

!pip install langdetect

from langdetect import detect 


lang_vec = []

i= 0

for caption in caption_string_vec:
    
    try :
        ans = detect(str(caption))
        lang_vec.append(ans)
    except :
        lang_vec.append("none")

# change lang

new_lang_vec = []

for lang in lang_vec:
    if lang in ["id", "en"]:
        new_lang_vec.append(lang)
    else:
        new_lang_vec.append("id")


    

data_with_string = raw_data

data_with_string["lang"] = new_lang_vec

data_with_string = raw_data

data_with_string["string_caption"] = caption_string_vec

data_with_string.head()
# change type

data_coded = data_with_string

data_coded["type"] = data_coded["type"].astype("string") 
data_coded["is_business_account"] = data_coded["is_business_account"].astype("string") 
data_coded["business_category_name"] = data_coded["business_category_name"].astype("string") 
data_coded["gender"] = data_coded["gender"].astype("string") 
data_coded["user_exposure"] = data_coded["user_exposure"].astype("string") 
data_coded["lang"] = data_coded["lang"].astype("string") 

print(data_coded.dtypes)
# coding

x = data_coded["business_category_name"][0]

data_coded["comments_disabled"] = data_coded["comments_disabled"].replace({False : 1, True : 0})
data_coded["is_video"] = data_coded["is_video"].replace({False : 0, True : 1})
data_coded["type"] = data_coded["type"].replace({"GraphImage" : 0, "GraphSidecar" : 1, "GraphVideo" : 2})
data_coded["has_external_url"] = data_coded["has_external_url"].replace({False : 0, True : 1})
data_coded["is_business_account"] = data_coded["is_business_account"].replace({"False" : 0, "True" : 1})
data_coded["business_category_name"] = data_coded["business_category_name"].replace({"Creators & Celebrities" : "1", "Local Events" : "0" , x : "-1"})
data_coded["gender"] = data_coded["gender"].replace({"Female" : "0", "Male" : "1"})
data_coded["user_exposure"] = data_coded["user_exposure"].replace({"National" : "0", "International" : "1"})
data_coded["lang"] = data_coded["lang"].replace({"en" : "1", "id" : "0"})

data_coded.head()


# First make a function to delete repetitive alphabet
import itertools

def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

import re

def remove_nonalphanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


def to_lower_case(text):
    return text.lower()


def preprocessing_text(text):
    text = remove_repeating_characters(text)
    text = remove_nonalphanumeric(text)
    text = to_lower_case(text)
    
    return text

# Check our function
preprocessing_text('Bagus\n\n\nNamun Akan Lebih Baik Apabila Lebih')
# Apply function to column 'string_function'

clean_caption_data = data_coded
clean_caption_data['captions'] = clean_caption_data['captions'].apply(lambda x: preprocessing_text(x))

clean_caption_data.head()
# Translate bahasa inggris ke indonesia

#!pip install translate

from translate import Translator

translator= Translator(to_lang="Indonesian")


def translate_to_id(row):
    
    text = row["captions"]
    
    if row["lang"] == 0:
        return text
    else:
        return translator.translate(text)

translated_data = clean_caption_data

translated_data["captions"] = translated_data.apply(lambda row: translate_to_id(row), axis = 1)
translated_data["type"] = data_coded["type"]
translated_data.head()
def get_len_caption(row):
    
    return( len(row["captions"]))

translated_data["len_caption"] = translated_data.apply(lambda row :  get_len_caption(row), axis = 1)

translated_data.head()
translated_data["is_business_account"] = data_coded["is_business_account"]

translated_data.head()
# Make a vector to contain all unique word in 'review sangat singkat'
translated_data['captions']
unique_string = set()
for x in translated_data['captions']:
    for y in x.split():
        unique_string.add(y)
        
len(unique_string)
#Count statistics of number of word in review

len_data = [len(x.split()) for x in translated_data['captions']]
print(np.mean(len_data))
print(np.median(len_data))
print(np.std(len_data))
print(np.min(len_data))
print(np.max(len_data))
print(np.percentile(len_data, 98))
embed_size = 100 # how big is each word vector
max_features = 23000 # how many unique words to use
maxlen = 20 # max number of words in a comment to use
# Example
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 4)
tokenizer.fit_on_texts(["ini sebuah kalimat hehehe"])
examples = tokenizer.texts_to_sequences(["ini contoh kalimat juga"])
print(examples[0])
# Real one

preprocessing_data = translated_data

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(preprocessing_data['captions'])
list_tokenized_train = tokenizer.texts_to_sequences(preprocessing_data['captions'].values)
list_tokenized_train[0]
# Example

from keras.preprocessing.sequence import pad_sequences
pad_sequences(examples, maxlen = maxlen)
# Real one

X_t = pad_sequences(list_tokenized_train, maxlen= maxlen )
X_t[0]
len(X_t) == len(translated_data)
clean_data = translated_data


clean_data["type"] = clean_data["type"].astype("int64") 
clean_data["is_business_account"] = clean_data["is_business_account"].astype("int64") 
clean_data["business_category_name"] = clean_data["business_category_name"].astype("int64") 
clean_data["gender"] = clean_data["gender"].astype("int64") 
clean_data["user_exposure"] = clean_data["user_exposure"].astype("int64") 
clean_data["lang"] = clean_data["lang"].astype("int64") 

clean_data = clean_data.drop(["captions"], axis = 1)
clean_data = clean_data.drop(["string_caption"], axis = 1)
clean_data = clean_data.drop(["engagement"], axis = 1)


df_list = clean_data.values.tolist()

df_list[0]
len(df_list[0])


# Create input list

X_new = []

for i in range(len(X_t)):
    
    y = df_list[i]  + X_t[i].tolist()
    
    X_new = X_new+[y]
    
    i = i +1
    
X_new_array = np.array(X_new)

X_new_array
len(X_new_array[0])

import gensim

DIR_DATA_MISC = "../input/word2vec-100-indonesian"
path = '{}/idwiki_word2vec_100.model'.format(DIR_DATA_MISC)
id_w2v = gensim.models.word2vec.Word2Vec.load(path)
print(id_w2v.most_similar('itb'))
maxlen
word_index = tokenizer.word_index
nb_words = max_features
embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
for word, i in word_index.items():
    cur = word
    if cur in index2word_set:
        embedding_matrix[i] = id_w2v[cur]
        continue
        
    embedding_matrix[i] = unknown_vector
# Import needed packages
# And make needed function


from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import callbacks

from keras import backend as K


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model():
    
    meta_input = Input(shape= (18,))
    inp = Input(shape=(maxlen,))
    
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(32, return_sequences=True))(x)
    x2 = Bidirectional(GRU(32, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    conc1 = Concatenate()([conc, meta_input])
    #print("\nSampai sini\n")
    #x = Dense(1, activation="sigmoid")(conc)
    x1 = Dense(256, activation = "relu")(conc1)
    #x15 = Dropout(0.5)(x1)
    x2 = Dense(256, activation = "relu")(x1)
    #x25 = Dropout(0.5)(x2)
    x3 = Dense(256, activation = "relu")(x2)
    x = Dense(1, activation="linear")(x3)
    #print("\nSampai sini woyy\n")
    
    model = Model(inputs= [meta_input, inp], outputs=x)
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
from sklearn.model_selection import KFold
def get_kfold():
    return KFold(n_splits=5, shuffle=True, random_state=1)

import tensorflow as tf

X = X_t
#X_add = np.array(df_list)
X_add = np.array(df_list_standardize)
#X = X_new_array
#y = translated_data["engagement"].values

y = y_result

pred_cv = np.zeros(len(y))
count = 0

for train_index, test_index in get_kfold().split(X,X_add, y):
    count += 1
    print(count, end='')
    X_train, X_test = X[train_index], X[test_index]
    X_add_train, X_add_test = X_add[train_index], X_add[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    es = callbacks.EarlyStopping(monitor='val_rmse', min_delta=0.0001, patience=8,
                                             verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)
    
    
    model = get_model()
    model.fit([X_add_train,X_train], 
             y_train, batch_size=16, epochs=4,
             validation_data=([X_add_test, X_test], y_test),
             callbacks=[es, rlr],
             verbose=1)
    
    pred_cv[[test_index]] += model.predict([X_add_test, X_test])[:,0]
standardize_data.columns
# Standardize

standardize_data = clean_data

standardize_data.head()

#from sklearn import preprocessing

# Create the Scaler object
#scaler = preprocessing.StandardScaler()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


"""
ct = ColumnTransformer([
        ('follow_scaler', StandardScaler(), ['followers', 'following'])
    ], remainder='passthrough')

standardize_data = ct.fit_transform(standardize_data)
"""

follow_scaler = StandardScaler()

df1 = standardize_data

follow_scaler.fit(df1)
df1 = follow_scaler.transform(df1)

standardize_result = pd.DataFrame(df1,  columns=standardize_data.columns)

standardize_result.head()
df_list_standardize = standardize_result.values.tolist()
y = raw_data["engagement"]

y_angle = np.arctan(y)

y_sin = np.sin(y_angle* np.pi / 180)

y_result = np.log(y)/np.log(2)

y_result
np.sin(0.78539816339744828)

caption_string_test_vec = []

# Subtitute emoji

import re


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


for caption in test_data["captions"]:
    
    caption_string_test_vec.append(str(deEmojify(caption)))
    
#caption_string_vec

# Detect language

!pip install langdetect

from langdetect import detect 


lang_test_vec = []

i= 0

for caption in caption_string_test_vec:
    
    try :
        ans = detect(str(caption))
        lang_test_vec.append(ans)
    except :
        lang_test_vec.append("none")

# change lang

new_lang_vec = []

for lang in lang_test_vec:
    if lang in ["id", "en"]:
        new_lang_vec.append(lang)
    else:
        new_lang_vec.append("id")


    

data_test_with_string = test_data

data_test_with_string["lang"] = new_lang_vec

#data_with_string = raw_data

data_test_with_string["string_caption"] = caption_string_test_vec

data_test_with_string.head()
#data_test_coded.head()
# change type

data_test_coded = data_test_with_string

data_test_coded["type"] = data_test_coded["type"].astype("string") 
data_test_coded["is_business_account"] = data_test_coded["is_business_account"].astype("string") 
data_test_coded["business_category_name"] = data_test_coded["business_category_name"].astype("string") 
data_test_coded["gender"] = data_test_coded["gender"].astype("string") 
data_test_coded["user_exposure"] = data_test_coded["user_exposure"].astype("string") 
data_test_coded["lang"] = data_test_coded["lang"].astype("string") 

print(data_test_coded.dtypes)

# coding

x = data_test_coded["business_category_name"][2]

data_test_coded["comments_disabled"] = data_test_coded["comments_disabled"].replace({False : 1, True : 0})
data_test_coded["is_video"] = data_test_coded["is_video"].replace({False : 0, True : 1})
data_test_coded["type"] = data_test_coded["type"].replace({"GraphImage" : 0, "GraphSidecar" : 1, "GraphVideo" : 2})
data_test_coded["has_external_url"] = data_test_coded["has_external_url"].replace({False : 0, True : 1})
data_test_coded["is_business_account"] = data_test_coded["is_business_account"].replace({"False" : 0, "True" : 1})
data_test_coded["business_category_name"] = data_test_coded["business_category_name"].replace({"Creators & Celebrities" : "1", "Local Events" : "0" , x : "-1"})
data_test_coded["gender"] = data_test_coded["gender"].replace({"Female" : "0", "Male" : "1"})
data_test_coded["user_exposure"] = data_test_coded["user_exposure"].replace({"National" : "0", "International" : "1"})
data_test_coded["lang"] = data_test_coded["lang"].replace({"en" : "1", "id" : "0"})

data_test_coded.head()
# Apply function to column 'string_function'

clean_caption_data_test = data_test_coded
clean_caption_data_test['captions'] = clean_caption_data_test['captions'].apply(lambda x: preprocessing_text(x))

clean_caption_data_test.head()
#clean_caption_data_test.head()
from translate import Translator

translator= Translator(to_lang="Indonesian")


def translate_to_id(row):
    
    text = row["captions"]
    
    if row["lang"] == 0:
        return text
    else:
        #return translator.translate(text)
        return text

translated_data_test = clean_caption_data_test

translated_data_test["captions"] = translated_data_test.apply(lambda row: translate_to_id(row), axis = 1)

def get_len_caption(row):
    
    return( len(row["captions"]))

translated_data_test["len_caption"] = translated_data_test.apply(lambda row :  get_len_caption(row), axis = 1)

translated_data_test.head()
translated_data_test["captions"][0]

#tokenizer.fit_on_texts(preprocessing_data['captions'])

preprocessing_data_test = translated_data_test

list_tokenized_train = tokenizer.texts_to_sequences(preprocessing_data_test['captions'].values)

X_t_test = pad_sequences(list_tokenized_train, maxlen= maxlen )


clean_data_test = translated_data_test

clean_data_test["type"] = clean_data_test["type"].astype("int64") 
clean_data_test["is_business_account"] = clean_data_test["is_business_account"].astype("int64") 
clean_data_test["business_category_name"] = clean_data_test["business_category_name"].astype("int64") 

clean_data_test["business_category_name"] = data_test_coded["business_category_name"]

clean_data_test["gender"] = clean_data_test["gender"].astype("int64") 
clean_data_test["user_exposure"] = clean_data_test["user_exposure"].astype("int64") 
clean_data_test["lang"] = clean_data_test["lang"].astype("int64") 

clean_data_test = clean_data_test.drop(["captions"], axis = 1)
clean_data_test = clean_data_test.drop(["string_caption"], axis = 1)




df_list_test = clean_data_test.values.tolist()


X_test_add = np.array(df_list_test)
standardize_test_data = clean_data_test

df1 = follow_scaler.transform(standardize_test_data)

standardize_result_test = pd.DataFrame(df1,  columns=clean_data_test.columns)

df_list_test_standardize = standardize_result_test.values.tolist()

df_list_test_standardize

X_test_add_standardize = np.array(df_list_test_standardize)
len(X_test_add[0])
model_test = model

"""
model_test.fit([X_add,X_t], 
             y, batch_size=16, epochs=4,
             validation_data=([X_add, X_t], y),
             callbacks=[es, rlr],
             verbose=1)
"""
#y_pred = model_test.predict([X_test_add, X_t_test])
#X_test_add_standardize
y_pred = model_test.predict([X_test_add_standardize, X_t_test])
y_pred
y_final = []


y_final = np.power(2, y_pred)

y_final
sub = pd.read_csv("../input/hmif-challenge-2019/sample-submission.csv")

sub.head()
sub["engagement"] = y_pred

sub.head()
submission = sub.to_csv("ans_1.csv", index = False)
