# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import gc



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding

from keras.layers import BatchNormalization

from keras.layers import LSTM

from keras.layers import Input

from keras.layers import Dropout

from keras.layers import Reshape

from keras.layers import Concatenate

from keras.layers import GRU

from keras.models import Model

from keras.layers import SpatialDropout1D

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from keras import backend as K



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/guess-my-price/train.tsv',sep='\t')

test = pd.read_csv('../input/guess-my-price/test.tsv',sep='\t')

train.drop(columns=['random'],inplace=True,axis=1)

test.drop(columns=['random'],inplace=True,axis=1)
train.dtypes
# Preprocess Dataframe

def preprocess_text(dataframe, ignore_words_list = []):

    for column in dataframe.columns:

        if dataframe[column].dtypes == 'object':

            

            print(column)

            

            #Replace nan

            dataframe[column].fillna('unavailable',inplace=True)

            

            #Lower string

            dataframe[column] = dataframe[column].str.lower()

            

            #Punctuation

            dataframe[column] = dataframe[column].str.replace(r'[^\w\s]+', ' ')



            #Remove non-ascii characters                

#             dataframe[column] = dataframe[column].apply(lambda x: ''.join([" " if i not in string.printable else i for i in str(x)]))



            #Remove numbers

            dataframe[column] = dataframe[column].str.replace(r'\d', '')



            #Replace Words

            dataframe[column] = dataframe[column].apply(lambda x: ' '.join([word for word in x.split() if word not in ignore_words_list]))

            

    return dataframe
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
test['price'] = -1

train_test_combined = train.append(test,sort=True).reset_index(drop=True)

train_test_combined.head()
train_test_combined = preprocess_text(train_test_combined, stop_words)

train_test_combined.head()
from keras.preprocessing.text import Tokenizer

# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(train_test_combined['name'])

name_word_index = t.word_index

name_txt_to_seq = t.texts_to_sequences(train_test_combined['name'])



t = Tokenizer()

t.fit_on_texts(train_test_combined['brand_name'])

brand_name_word_index = t.word_index

brand_name_txt_to_seq = t.texts_to_sequences(train_test_combined['brand_name'])



t = Tokenizer()

t.fit_on_texts(train_test_combined['item_description'])

itm_desc_word_index = t.word_index

itm_desc_txt_to_seq = t.texts_to_sequences(train_test_combined['item_description'])



t = Tokenizer()

t.fit_on_texts(train_test_combined['category_name'])

cat_0_word_index = t.word_index

cat_0_txt_to_seq = t.texts_to_sequences(train_test_combined['category_name'])
# Max Length of Sentences

def FindMaxLength(lst): 

    maxList = max(lst, key = lambda i: len(i)) 

    maxLength = len(maxList) 

    return maxList, maxLength 



# get the maximum length of array from above array of arrays

# input length

max_length_name = FindMaxLength(name_txt_to_seq)[1]

max_length_brand_name = FindMaxLength(brand_name_txt_to_seq)[1]

max_length_itm_desc = FindMaxLength(itm_desc_txt_to_seq)[1]

max_length_cat_0 = FindMaxLength(cat_0_txt_to_seq)[1]



from keras.preprocessing.sequence import pad_sequences

# Pad the arrays with smaller value than max length with trailing zeros

padded_docs_name = pad_sequences(name_txt_to_seq, maxlen=max_length_name, padding='post')

padded_docs_brand_name = pad_sequences(brand_name_txt_to_seq, maxlen=max_length_brand_name, padding='post')

padded_docs_itm_desc = pad_sequences(itm_desc_txt_to_seq, maxlen=max_length_itm_desc, padding='post')

padded_docs_cat_0 = pad_sequences(cat_0_txt_to_seq, maxlen=max_length_cat_0, padding='post')
del name_txt_to_seq

del brand_name_txt_to_seq

del cat_0_txt_to_seq

del itm_desc_txt_to_seq

del train_test_combined

gc.collect()
embeddings_index = {}

f = open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.25d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# Generating Embedding Matrix

embedding_matrix_name = np.zeros((len(name_word_index) + 1, 25))

for word, i in name_word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_name[i] = embedding_vector

        

embedding_matrix_brand_name = np.zeros((len(brand_name_word_index) + 1, 25))

for word, i in brand_name_word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_brand_name[i] = embedding_vector

        

embedding_matrix_item_desc = np.zeros((len(itm_desc_word_index) + 1, 25))

for word, i in itm_desc_word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_item_desc[i] = embedding_vector

        

embedding_matrix_cat_0 = np.zeros((len(cat_0_word_index) + 1, 25))

for word, i in cat_0_word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_cat_0[i] = embedding_vector
del embeddings_index

gc.collect()
def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
def build_embedding_network():

    

    inputs = []

    embeddings = []

    

    # Embedding for name column

    vocab_size_name = len(name_word_index) + 1

    input_name = Input(shape=(max_length_name,))

#     embedding = Embedding(vocab_size_name, 25, weights=[embedding_matrix_name], input_length=max_length_name, name='name_embed', trainable=False)(input_name)

    embedding = Embedding(vocab_size_name, 25, input_length=max_length_name, name='name_embed', trainable=True)(input_name)

    embedding = SpatialDropout1D(0.15)(embedding)

    lstm = LSTM(25, dropout=0.2, recurrent_dropout=0.2)(embedding)

#     embedding = Reshape(target_shape=(25*max_length_name,))(lstm)

    inputs.append(input_name)

    embeddings.append(lstm)

    

    # Embedding for brand_name column

    vocab_size_brand_name = len(brand_name_word_index) + 1

    input_brand_name = Input(shape=(max_length_brand_name,))

#     embedding = Embedding(vocab_size_brand_name, 25, weights=[embedding_matrix_brand_name], input_length=max_length_brand_name, name='brand_name_embed', trainable=False)(input_brand_name)

    embedding = Embedding(vocab_size_brand_name, 25, input_length=max_length_brand_name, name='brand_name_embed', trainable=True)(input_brand_name)

#     lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(embedding)

    embedding = SpatialDropout1D(0.15)(embedding)

    embedding = Reshape(target_shape=(25*max_length_brand_name,))(embedding)

    inputs.append(input_brand_name)

    embeddings.append(embedding)

    

    vocab_size_itm_desc = len(itm_desc_word_index) + 1

    input_item_desc = Input(shape=(max_length_itm_desc,))

#     embedding = Embedding(vocab_size_itm_desc, 25, weights=[embedding_matrix_item_desc], input_length=max_length_itm_desc, name='item_desc_embed', trainable=False)(input_item_desc)

    embedding = Embedding(vocab_size_itm_desc, 25, input_length=max_length_itm_desc, name='item_desc_embed', trainable=True)(input_item_desc)

    embedding = SpatialDropout1D(0.15)(embedding)

    lstm = LSTM(25, dropout=0.2, recurrent_dropout=0.2)(embedding)

#     embedding = Reshape(target_shape=(25*max_length_itm_desc,))(lstm)

    inputs.append(input_item_desc)

    embeddings.append(lstm)



    # Embedding for category_0 column

    vocab_size_cat_0 = len(cat_0_word_index) + 1

    input_cat0 = Input(shape=(max_length_cat_0,))

#     embedding = Embedding(vocab_size_cat_0, 25, weights=[embedding_matrix_cat_0], input_length=max_length_cat_0, name='cat0_embed', trainable=False)(input_cat0)

    embedding = Embedding(vocab_size_cat_0, 25, input_length=max_length_cat_0, name='cat0_embed', trainable=True)(input_cat0)

    embedding = SpatialDropout1D(0.15)(embedding)

#     lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(embedding)

    embedding = Reshape(target_shape=(25*max_length_cat_0,))(embedding)

    inputs.append(input_cat0)

    embeddings.append(embedding)

    

#     # Embedding for item_condition column

#     vocab_size = train_x['item_condition_id'].nunique() + 1

#     input_item_condition = Input(shape=(1,))

#     embedding = Embedding(vocab_size, 1, input_length=1, name='itm_condition_id_embed', trainable=False)(input_item_condition)

#     embedding = Dropout(0.20)(embedding)

#     embedding = Reshape(target_shape=(3*1,))(embedding)

#     inputs.append(input_item_condition)

#     embeddings.append(embedding)



#     # Embedding for shipping column

#     vocab_size = train_x['shipping'].nunique() + 1             # no of unique words in train_

#     input_shipping = Input(shape=(1,))

#     embedding = Embedding(vocab_size, 1, input_length=1, name='shipping_embed', trainable=False)(input_shipping)

#     embedding = Dropout(0.20)(embedding)

#     embedding = Reshape(target_shape=(2*1,))(embedding)

#     inputs.append(input_shipping)

#     embeddings.append(embedding)

    

    # Dense layer for continous variables for item description column

    input_item_desc_senti = Input(shape=(202,))

    bn0 = BatchNormalization()(input_item_desc_senti)

    numeric = Dense(200, activation='relu')(bn0) #30

    numeric = Dropout(.35)(numeric)

    numeric = Dense(100)(numeric)

    numeric = Dropout(.20)(numeric)

    numeric = Dense(50)(numeric)

    inputs.append(input_item_desc_senti)

    embeddings.append(numeric)

    

    x = Concatenate()(embeddings)

    

    bn1 = BatchNormalization()(x)

    x = Dense(500, activation='relu')(bn1)

    x = Dropout(.35)(x)

    bn2 = BatchNormalization()(x)

    x = Dense(200, activation='relu')(bn2)

    x = Dropout(.2)(x)

    bn3 = BatchNormalization()(x)

    x = Dense(100, activation='relu')(bn3)

    x = Dropout(.08)(x)

    bn4 = BatchNormalization()(x)

    x = Dense(50, activation='relu')(bn4)

    output = Dense(1, activation='linear')(x)   

    model = Model(inputs, output)

    model.compile(loss = root_mean_squared_error, optimizer = "rmsprop")    

    print(model.summary())

    return model
glove_train = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_train_universe.csv")

glove_test = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_test_universe.csv")

glove_train.columns
cont_cols = [col for col in glove_train.columns if col not in ['train_id','price']]
%%time



input_list_train = []

input_list_val = []

input_list_test = []



                       # TRAIN DATA

######################################################################

input_list_train.append(padded_docs_name[:1100000])

input_list_train.append(padded_docs_brand_name[:1100000])

input_list_train.append(padded_docs_itm_desc[:1100000])

input_list_train.append(padded_docs_cat_0[:1100000])



# Numerical features

input_list_train.append(glove_train[cont_cols][:1100000])

y_train = np.log(train[['price']].iloc[:1100000].values+1)





######################################################################





                       # VAL DATA

######################################################################

input_list_val.append(padded_docs_name[1100000:train.shape[0]])

input_list_val.append(padded_docs_brand_name[1100000:train.shape[0]])

input_list_val.append(padded_docs_itm_desc[1100000:train.shape[0]])

input_list_val.append(padded_docs_cat_0[1100000:train.shape[0]])



# Numerical features

input_list_val.append(glove_train[cont_cols][1100000:train.shape[0]])

y_val = np.log(train[['price']].iloc[1100000:1186585].values+1)

#######################################################################





                       # TEST DATA

#######################################################################

input_list_test.append(padded_docs_name[train.shape[0]:])

input_list_test.append(padded_docs_brand_name[train.shape[0]:])

input_list_test.append(padded_docs_itm_desc[train.shape[0]:])

input_list_test.append(padded_docs_cat_0[train.shape[0]:])



# Numerical features

input_list_test.append(glove_test[cont_cols])
del padded_docs_name

del padded_docs_brand_name

del padded_docs_itm_desc

del padded_docs_cat_0

del train

# del test

gc.collect()
model = build_embedding_network()
import os



file_path = "/kaggle/output"

directory = os.path.dirname(file_path)

print(file_path)

try:

    os.stat(file_path)

except:

    os.mkdir(file_path) 
from glob import glob

glob('/kaggle/*')
filepath=file_path+"/model.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=2, min_lr=0.001,  verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
#preprocessing

history = model.fit(input_list_train,y_train, 

                    validation_data=(input_list_val,y_val), 

                    epochs=6,

                    callbacks=callbacks_list, 

                    shuffle=False, 

                    batch_size=512,

                    verbose=1)
# load the model

from keras.models import Sequential, load_model

new_model = load_model(filepath, custom_objects={'root_mean_squared_error': root_mean_squared_error})
predictions = new_model.predict(input_list_test,verbose=1)
submission = pd.DataFrame({'train_id':test.train_id})

submission['price'] = np.exp(predictions)-1

submission.to_csv('submission.csv', index=False)
submission.head(20)