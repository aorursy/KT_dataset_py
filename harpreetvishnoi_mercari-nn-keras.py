# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from string import punctuation
from collections import Counter
import re
import numpy as np
from time import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_log_error
# Any results you write to the current directory are saved as output.

####KERAS####
from numpy import array,asarray
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
train=pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv',sep='\t')
test=pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv',sep='\t')
train.category_name.fillna(value="missing", inplace=True)
train.brand_name.fillna(value="missing", inplace=True)
train.item_description.fillna(value="missing", inplace=True)
train.item_description.replace('No description yet',"missing", inplace=True)

test.category_name.fillna(value="missing", inplace=True)
test.brand_name.fillna(value="missing", inplace=True)
test.item_description.fillna(value="missing", inplace=True)
test.item_description.replace('No description yet',"missing", inplace=True)

print(train.shape)
train.head()
print(test.shape)
test.head()
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train['general_cat']=train['category_name'][0].split("/")[0]
train['subcat_1']=train['category_name'][0].split("/")[1]
train['subcat_2']=train['category_name'][0].split("/")[2]
test['general_cat']=test['category_name'][0].split("/")[0]
test['subcat_1']=test['category_name'][0].split("/")[1]
test['subcat_2']=test['category_name'][0].split("/")[2]

print("Processing categorical data...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(list(train.brand_name)+list(test.brand_name))
#train['brand_name_le'] = le.transform(train.brand_name)
#test['brand_name_le'] = le.transform(test.brand_name)
train['brand_name'] = le.transform(train.brand_name)
test['brand_name'] = le.transform(test.brand_name)

le.fit(np.hstack([train.category_name, test.category_name]))#remove this to use 3 seperate types later
train.category_name = le.transform(train.category_name)
test.category_name = le.transform(test.category_name)

le.fit(list(train.general_cat)+list(test.general_cat))
train['general_cat_le'] = le.transform(train.general_cat)
test['general_cat_le'] = le.transform(test.general_cat)

le.fit(list(train.subcat_1)+list(test.subcat_1))
train['subcat_1_le'] = le.transform(train.subcat_1)
test['subcat_1_le'] = le.transform(test.subcat_1)

le.fit(list(train.subcat_2)+list(test.subcat_2))
train['subcat_2_le'] = le.transform(train.subcat_2)
test['subcat_2_le'] = le.transform(test.subcat_2)
t0=time()
t=Tokenizer()
raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])#Try to add test set also

print("   Fitting tokenizer...")
t.fit_on_texts(raw_text)
print("   Transforming text to seq...")

train["seq_item_description"] = t.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = t.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = t.texts_to_sequences(train.name.str.lower())
test["seq_name"] = t.texts_to_sequences(test.name.str.lower())
print(time()-t0)
train.head(3)

max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x))), np.max(test.seq_item_description.apply(lambda x: len(x)))])
print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_seq_item_description))
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75#try 100
MAX_TEXT = np.max([np.max(train.seq_name.max()),np.max(test.seq_name.max()), np.max(train.seq_item_description.max()),np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()])+1
MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1
#SCALE target variable
train["target"] = np.log(train.price+1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.reshape(-1,1))
pd.DataFrame(train.target).hist()
dtrain, dvalid = train_test_split(train, random_state=123, test_size=0.01)
print(dtrain.shape)
print(dvalid.shape)
#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name':            pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ,padding='post')#multiple interger values
        ,'item_desc':      pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ,padding='post')#multiple interger values
        ,'brand_name':     np.array(dataset.brand_name)
        ,'category_name':  np.array(dataset.category_name)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars':       np.array(dataset[["shipping"]])
    }
    return X

### Because of padding less then the maximim number of character which we have done above, the padded interger cuts the unwanted long sentences
###to make every variable of same length.

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")# Stop training when a monitored quantity has stopped improving.
    msave = ModelCheckpoint(filepath, save_best_only=True)#saves the best latest model and doesnt let it be overwritten
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))
def get_model():

    dr_r = 0.1
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    
    #Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)#represents a deeper relationship between words.#like gender and age relationship betweeen them
                                            #which in turn helps rnn to better classify and adjust them.
                                            #also weights are randomly adjusted. Can use pretrained maybe.
            
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    #rnn layer
    #rnn_layer1 = GRU(16) (emb_item_desc)#LSTM can be used
    #rnn_layer2 = GRU(8) (emb_name)
    
    #main layer
    main_l = concatenate([
                            Flatten() (emb_brand_name)
                            , Flatten() (emb_category_name)
                            , Flatten() (emb_item_condition)
                            , GRU(16)   (emb_item_desc)
                            , GRU(8)     (emb_name)
                            , num_vars
                          ])
    
    main_l=Dense(128)(main_l)
    main_l=Dropout(dr_r)(main_l)
    
    main_l=Dense(128)(main_l)
    main_l=Dropout(dr_r)(main_l)
    
    #main_l = Dropout(dr_r) (Dense(128) (main_l))
    #main_l = Dropout(dr_r) (Dense(64) (main_l))
    
    #output
    main_l = Dense(1) (main_l)
    output = Activation('linear')(main_l)
    
    #output = Dense(1, activation="linear") (main_l)
    
    
    
    #model
    model = Model([name, item_desc, brand_name,category_name, item_condition, num_vars], output)
    
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    #model.compile(loss="mse", optimizer="adam", metrics=[rmsle])#change this
    
    return model

    
model = get_model()
model.summary()
BATCH_SIZE = 40000
epochs = 1

model = get_model()
model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_valid, dvalid.target), verbose=1)
# model.save_weights('model_weight.h5')

# model_json=model.to_json()
# with open('model.json','w') as json_file:
#     json_file.write(model_json)
import math
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

# #EVLUEATE THE MODEL ON DEV TEST: What is it doing?
# val_preds = model.predict(X_valid)
# val_preds = target_scaler.inverse_transform(val_preds)
# val_preds = np.exp(val_preds)+1

# #mean_absolute_error, mean_squared_log_error
# y_true = np.array(dvalid.price.values)
# y_pred = val_preds[:,0]
# v_rmsle = rmsle(y_true, y_pred)
# print(" RMSLE error on dev test: "+str(v_rmsle))


############################




# df_output=pd.DataFrame()
# df_output['test_id']=test['test_id']
# ytest_pred=rf.predict(xtest)
# df_output['price']=ytest_pred
# df_output.to_csv("Submission_1.csv",index=False)


