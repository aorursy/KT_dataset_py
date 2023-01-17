# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

import gc

import os

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

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

from sklearn.model_selection import KFold

from keras.models import Sequential, load_model

from tensorflow.keras import backend as K



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
train = train.drop(train[train.price <= 1.0].index).reset_index(drop=True)
def to_lower(dataset):

    

    dataset['category_name'] = dataset['category_name'].str.lower()

    dataset['brand_name'] = dataset['brand_name'].str.lower()

    dataset['brand_name'] = dataset['brand_name'].str.replace("'","")

    dataset['name'] = dataset['name'].str.lower()

    dataset['name'] = dataset['name'].str.replace("'","")

    dataset['item_description'] = dataset['item_description'].str.lower()

    

def fill_brands(dataset, test):

    brands = pd.concat([dataset['brand_name'], test['brand_name']], axis=0).unique().astype(str)

    print(pd.isnull(dataset['brand_name']).sum())

    brands_str = re.compile(r'\b(?:%s)\b' % '|'.join(brands))

    dataset['brand_name'] = dataset.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)

    test['brand_name'] = test.apply(lambda row: row['brand_name'] if pd.notnull(row['brand_name']) or brands_str.match(row['name']) is None else brands_str.match(row['name']).group(0), axis=1)

    print(pd.isnull(dataset['brand_name']).sum())

    del brands

    del brands_str

    gc.collect()

    

to_lower(train)

to_lower(test)

fill_brands(train, test)
# def preprocess_dataframe(df):

#     df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')

#     df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))

#     return df[['name', 'text', 'shipping', 'item_condition_id']]
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

#             dataframe[column] = dataframe[column].str.replace(r'[^\w\s]+', ' ')

            dataframe[column] = dataframe[column].str.replace(r'/', ' ')

            dataframe[column] = dataframe[column].str.replace(r'!', ' ')

            dataframe[column] = dataframe[column].str.replace(r'&', ' ')

            dataframe[column] = dataframe[column].str.replace(r'(', ' ')

            dataframe[column] = dataframe[column].str.replace(r')', ' ')

            

            #Replace values

            to_replace = [

            r'(\d+)(\.)(\d+)',      # 1.5 ml -> 1`5 ml

            r'(\d+)(\s+)?[gG][bB]?\s+', # 16 gb, 16GB, 16 g -> 16g

            r'(\d+)(\s+)?[tT][bB]?\s+', # 1 tb, 1TB, 1 T -> 1t

            #r'(\d+(`\d+)?)\s+[mM][lL][\s\W]+', # "5`32 ml. 5 ml. 5`0 ml." -> 5`32ml

            #r'(\d+(`\d+)?)(\s+)?([fF][lL](uit)?)?(\s+)?[oO][zZ][\s\W]+', # 5`32 fluit oz/4 fl oz/5`0 oz/8fl oz -> 5`32oz

            r'\s+[tT][\s+|-][Ss][hH][iI][rR][tT]', # t shirt/t-shirt -> tshirt

            r'(\d+)kt\s+', # 14kt -> 14k; this is for Jewelry products

            r'\s+S925\s+', # S925 -> 925; this is for Jewelry products

            ]

            value = [

            r'\1`\3',

            r'\1g ',

            r'\1t ',

            #r'\1ml ',

            #r'\1oz ',

            ' tshirt',

            r'\1k ',

            r' 925 ',

            ]

            dataframe[column] = dataframe[column].replace(to_replace=to_replace, value=value, regex=True)



            #Remove non-ascii characters                

#             dataframe[column] = dataframe[column].apply(lambda x: ''.join([" " if i not in string.printable else i for i in str(x)]))



            #Remove numbers

#             dataframe[column] = dataframe[column].str.replace(r'\d', '')



            #Replace Words

#             dataframe[column] = dataframe[column].apply(lambda x: ' '.join([word for word in x.split() if word not in ignore_words_list]))

            

    return dataframe
from nltk.corpus import stopwords

stop_words = stopwords.words('english')



test['price'] = -1

train_test_combined = train.append(test,sort=True).reset_index(drop=True)



# Preprocess Dataframe

# train_test_combined = preprocess_dataframe(train_test_combined)



#Preprocess Text

train_test_combined = preprocess_text(train_test_combined, stop_words)

train_test_combined.head(20)
def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true)))



def create_model(categorical_cols, numerical_cols, word_index, max_length):

    

    inputs = []

    embeddings = []

    

    # Embedding for categorical columns:

    for col in categorical_cols:

        vocab_size = len(word_index[col]) + 1

        input_cat_cols = Input(shape=(max_length[col],))

        embed_size = int(min(np.ceil((vocab_size)/2), 30)) #25

        embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding'.format(col), trainable=True)(input_cat_cols)

        embedding = SpatialDropout1D(0.20)(embedding)

        embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)

        inputs.append(input_cat_cols)

        embeddings.append(embedding)

    

    # Dense layer for continous variables for item description column

    input_num_cols = Input(shape=(len(numerical_cols),))

    bn0 = BatchNormalization()(input_num_cols)

    numeric = Dense(150, activation='relu')(bn0) #30

    numeric = Dropout(.20)(numeric)

    numeric = Dense(90)(numeric)

    inputs.append(input_num_cols)

    embeddings.append(numeric)

    

    x = Concatenate()(embeddings)

    

    bn1 = BatchNormalization()(x)

    x = Dense(500, activation='relu')(bn1)

    x = Dropout(.50)(x)

    bn2 = BatchNormalization()(x)

    x = Dense(200, activation='relu')(bn2)

    x = Dropout(.40)(x)

    bn3 = BatchNormalization()(x)

    x = Dense(100, activation='relu')(bn3)

    bn4 = BatchNormalization()(x)

    x = Dense(50, activation='relu')(bn4)

    output = Dense(1, activation='linear')(x)   

    model = Model(inputs, output)

    model.compile(loss = root_mean_squared_error, optimizer = "rmsprop")    

#     print(model.summary())

    return model
cat_cols = ['name', 'brand_name', 'item_description', 'category_name', 'item_condition_id']

# cat_cols = ['name', 'text', 'shipping', 'item_condition_id']



def get_padding(data, categorical_cols):

    # Tokenize Sentences

    word_index, max_length, padded_docs = {},{},{}

    for col in cat_cols:

        print("Processing column:", col)

        t = Tokenizer()

        t.fit_on_texts(data[col].astype(str))

        word_index[col] = t.word_index

        txt_to_seq = t.texts_to_sequences(data[col].astype(str))

        max_length[col] = len(max(txt_to_seq, key = lambda x: len(x)))

        padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')

    

    return word_index, max_length, padded_docs

    

word_index, max_length, padded_docs = get_padding(train_test_combined, cat_cols)
glove_train = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_train_universe.csv")

glove_test = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_test_universe.csv")

# ridge_train = pd.read_csv(r"/kaggle/input/tf-idf-ridge-regression-stacking/ridge_regression_train.csv")

# ridge_test = pd.read_csv(r"/kaggle/input/tf-idf-ridge-regression-stacking/ridge_regression_test.csv")



# glove_train = glove_train.merge(ridge_train, how = 'left', on = 'train_id')

# glove_test = glove_test.merge(ridge_test, how = 'left', on = 'train_id')



# del ridge_train, ridge_test

# gc.collect()



glove_train = glove_train.drop(glove_train[glove_train.price <= 1.0].index).reset_index(drop=True)



glove_train.drop(columns=['train_id','price',],inplace=True,axis=1)

glove_test.drop(columns=['train_id'],inplace=True,axis=1)

glove_train.columns
num_cols = glove_train.columns.to_list()
splits = 25



file_path = "/kaggle/output"

directory = os.path.dirname(file_path)

print(file_path)

try:

    os.stat(file_path)

except:

    os.mkdir(file_path) 

    

from glob import glob

glob('/kaggle/*')



test_preds = np.zeros((test.shape[0],1))

kf = KFold(n_splits=splits)

fold_number = 1

for train_index, validation_index in kf.split(train.index, train.price):

    print("Fold Number:",fold_number)

    input_list_train, input_list_val, input_list_test = [], [], []

    for col in cat_cols:

        input_list_train.append(padded_docs[col][train_index])

        input_list_val.append(padded_docs[col][validation_index])

        input_list_test.append(padded_docs[col][train.shape[0]:])

    input_list_train.append(glove_train.iloc[train_index])

    input_list_val.append(glove_train.iloc[validation_index])

    input_list_test.append(glove_test)

    

    y_train = np.log1p(train[['price']].iloc[train_index].values) # np.log(train[['price']].iloc[train_index].values+1) 

    y_val = np.log1p(train[['price']].iloc[validation_index].values) # np.log(train[['price']].iloc[validation_index].values+1)

    

    model = create_model(cat_cols, num_cols, word_index, max_length)

    if fold_number == 1:

        print(model.summary())

    filepath=file_path+"/model.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-6,  verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

    callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]

    entity_embedding_model = model.fit(input_list_train,y_train, 

                             validation_data=(input_list_val,y_val), 

                             epochs=30,#25

                             callbacks=callbacks_list, 

                             shuffle=False, 

                             batch_size=1024,

                             verbose=1)

    

    # Predictions

    new_model = load_model(filepath, custom_objects={'root_mean_squared_error': root_mean_squared_error})

    predictions = new_model.predict(input_list_test,verbose=1)

    test_preds += predictions

    

    K.clear_session()

    fold_number += 1



test_preds /= splits

submission = pd.DataFrame({'train_id':test.train_id})

submission['price'] = np.expm1(test_preds) # np.exp(test_preds)-1

submission.to_csv('submission.csv', index=False)
submission