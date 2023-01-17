import pandas as pd 
import numpy as np 
import time 
import gc 
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
cores = 4

from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) 
def preprocess_dataset(dataset):
    
    t1 = time.time()
    print("Filling Missing Values.....")
    dataset['price'] = dataset['price'].fillna(0).astype('float32')
    dataset['param_1'].fillna(value='missing', inplace=True)
    dataset['param_2'].fillna(value='missing', inplace=True)
    dataset['param_3'].fillna(value='missing', inplace=True)
    
    dataset['param_1'] = dataset['param_1'].astype(str)
    dataset['param_2'] = dataset['param_2'].astype(str)
    dataset['param_3'] = dataset['param_3'].astype(str)
    
    print("Casting data types to type Category.......")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')

    print("Creating New Feature.....")
    dataset['param123'] = (dataset['param_1']+'_'+dataset['param_2']+'_'+dataset['param_3']).astype(str)
    del dataset['param_2'], dataset['param_3']
    gc.collect()
        
    print("PreProcessing Function completed.")
    
    return dataset

def keras_fit(train):
    
    t1 = time.time()
    train['title_description']= (train['title']+" "+train['description']).astype(str)
    del train['description'], train['title']
    gc.collect()
    
    print("Start Tokenization.....")
    tokenizer = text.Tokenizer(num_words = max_words_title_description)
    all_text = np.hstack([train['title_description'].str.lower()])
    tokenizer.fit_on_texts(all_text)
    del all_text
    del train['activation_date']
    gc.collect()
    
    print("Loading Test for Label Encoding on Train + Test")
    use_cols_test = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3']
    test = pd.read_csv("../input/avito-demand-prediction/test.csv", usecols = use_cols_test)
    
    test['param_1'].fillna(value='missing', inplace=True)
    test['param_1'] = test['param_1'].astype(str)
    test['param_2'].fillna(value='missing', inplace=True)
    test['param_2'] = test['param_2'].astype(str)
    test['param_3'].fillna(value='missing', inplace=True)
    test['param_3'] = test['param_3'].astype(str)

    print("Creating New Feature.....")
    test['param123'] = (test['param_1']+'_'+test['param_2']+'_'+test['param_3']).astype(str)
    del test['param_2'], test['param_3']
    gc.collect()
    
    ntrain = train.shape[0]
    DF = pd.concat([train, test], axis = 0)
    del train, test
    gc.collect()
    print(DF.shape)
    
    print("Start Label Encoding process....")
    le_region = LabelEncoder()
    le_region.fit(DF.region)
    
    le_city = LabelEncoder()
    le_city.fit(DF.city)
    
    le_category_name = LabelEncoder()
    le_category_name.fit(DF.category_name)
    
    le_parent_category_name = LabelEncoder()
    le_parent_category_name.fit(DF.parent_category_name)
    
    le_param_1 = LabelEncoder()
    le_param_1.fit(DF.param_1)
    
    le_param123 = LabelEncoder()
    le_param123.fit(DF.param123)

    train = DF[0:ntrain]
    test = DF[ntrain:]
    del DF 
    gc.collect()
    
    train['price'] = np.log1p(train['price'])
    train['item_seq_number'] = np.log(train['item_seq_number'])
    print("Fit on Train Function completed.")
    
    return train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name, le_param_1, le_param123

def keras_train_transform(dataset):
    
    t1 = time.time()
    
    dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    del train['title_description']
    gc.collect()

    dataset['region'] = le_region.transform(dataset['region'])
    dataset['city'] = le_city.transform(dataset['city'])
    dataset['category_name'] = le_category_name.transform(dataset['category_name'])
    dataset['parent_category_name'] = le_parent_category_name.transform(dataset['parent_category_name'])
    dataset['param_1'] = le_param_1.transform(dataset['param_1'])
    dataset['param123'] = le_param123.transform(dataset['param123'])
    
    print("Transform on test function completed.")
    
    return dataset
    
def keras_test_transform(dataset):
    
    t1 = time.time()
    dataset['title_description']= (dataset['title']+" "+dataset['description']).astype(str)
    del dataset['description'], dataset['title']
    gc.collect()
    
    dataset['seq_title_description']= tokenizer.texts_to_sequences(dataset.title_description.str.lower())
    print("Transform done for test")
    print("Time taken for Sequence Tokens is"+str(time.time()-t1))
    
    del dataset['activation_date'], dataset['title_description']
    gc.collect()

    dataset['region'] = le_region.transform(dataset['region'])
    dataset['city'] = le_city.transform(dataset['city'])
    dataset['category_name'] = le_category_name.transform(dataset['category_name'])
    dataset['parent_category_name'] = le_parent_category_name.transform(dataset['parent_category_name'])
    dataset['param_1'] = le_param_1.transform(dataset['param_1'])
    dataset['param123'] = le_param123.transform(dataset['param123'])
    
    dataset['price'] = np.log1p(dataset['price'])
    dataset['item_seq_number'] = np.log(dataset['item_seq_number'])
    
    print("Transform on test function completed.")
    
    return dataset
    
def get_keras_data(dataset):
    X = {
        'seq_title_description': pad_sequences(dataset.seq_title_description, maxlen=max_seq_title_description_length)
        ,'region': np.array(dataset.region)
        ,'city': np.array(dataset.city)
        ,'category_name': np.array(dataset.category_name)
        ,'parent_category_name': np.array(dataset.parent_category_name)
        ,'param_1': np.array(dataset.param_1)
        ,'param123': np.array(dataset.param123)
        ,'price': np.array(dataset[["price"]])
        ,'item_seq_number': np.array(dataset[["item_seq_number"]])
    }
    
    print("Data ready for Vectorization")
    
    return X
# Loading Train data - No Params, No Image data 
dtypes_train = {
                'price': 'float32',
                'deal probability': 'float32',
                'item_seq_number': 'uint32'
}

# No user_id
use_cols = ['item_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'activation_date', 'deal_probability']
train = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)

y_train = train['deal_probability']
del train['deal_probability']
gc.collect()

max_seq_title_description_length = 100
max_words_title_description = 200000

train = preprocess_dataset(train)
train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name, le_param_1, le_param123 = keras_fit(train)
train = keras_train_transform(train)
print("Tokenization done and TRAIN READY FOR Validation splitting")

# Calculation of max values for Categorical fields 
        
max_region = np.max(train.region.max())+1
max_city= np.max(train.city.max())+1
max_category_name = np.max(train.category_name.max())+1
max_parent_category_name = np.max(train.parent_category_name.max())+1
max_param_1 = np.max(train.param_1.max())+1
max_param123 = np.max(train.param123.max())+1
    
print("Train Test Split")
x_train_f, x_valid_f, y_train_f, y_valid_f = train_test_split(train, y_train, train_size=0.95, random_state=233)
print(x_train_f.shape, x_valid_f.shape)
print(y_train_f.shape, y_valid_f.shape)

del train, y_train
gc.collect()

X_train = get_keras_data(x_train_f)
X_valid = get_keras_data(x_valid_f)
del x_train_f, x_valid_f
gc.collect()

# EMBEDDINGS COMBINATION 
# FASTTEXT

EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+1
EMBEDDING_DIM1 = 300# this is from the pretrained vectors
embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
print(embedding_matrix1.shape)
# Creating Embedding matrix 
c = 0 
c1 = 0 
w_Y = []
w_No = []
for word, i in tokenizer.word_index.items():
    if word in embeddings_index1:
        c +=1
        embedding_vector = embeddings_index1[word]
        w_Y.append(word)
    else:
        embedding_vector = None
        w_No.append(word)
        c1 +=1
    if embedding_vector is not None:    
        embedding_matrix1[i] = embedding_vector

print(c,c1, len(w_No), len(w_Y))
print(embedding_matrix1.shape)
del embeddings_index1
gc.collect()

print(" FAST TEXT DONE")

def RNN_model():

    #Inputs
    seq_title_description = Input(shape=[X_train["seq_title_description"].shape[1]], name="seq_title_description")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape = [1], name = 'item_seq_number')
    
    #Embeddings layers

    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = True)(seq_title_description)
    emb_region = Embedding(max_region, 10)(region)
    emb_city = Embedding(max_city, 10)(city)
    emb_category_name = Embedding(max_category_name, 10)(category_name)
    emb_parent_category_name = Embedding(max_parent_category_name, 10)(parent_category_name)
    emb_param_1 = Embedding(max_param_1, 10)(param_1)
    emb_param123 = Embedding(max_param123, 10)(param123)

    rnn_layer1 = GRU(25) (emb_seq_title_description)
    
    #main layer
    main_l = concatenate([
          rnn_layer1
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , Flatten() (emb_param_1)
        , Flatten() (emb_param123)
        , price
        , item_seq_number
    ])
    
    main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="sigmoid") (main_l)
    
    #model
    model = Model([seq_title_description, region, city, category_name, parent_category_name, param_1, param123, price, item_seq_number ], output)
    model.compile(optimizer = 'adam',
                  loss= root_mean_squared_error,
                  metrics = [root_mean_squared_error])
    return model

def rmse(y, y_pred):

    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 

def eval_model(model):
    val_preds = model.predict(X_valid)
    y_pred = val_preds[:, 0]
    
    y_true = np.array(y_valid_f)
    
    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)
    
    print(yt.isnull().any())
    print(yp.isnull().any())
    
    v_rmse = rmse(y_true, y_pred)
    print(" RMSE for VALIDATION SET: "+str(v_rmse))
    return v_rmse

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
from keras import optimizers

epochs = 1
BATCH_SIZE = 512 * 3
steps = int(len(X_train['seq_title_description'])/BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.0045
lr_decay = exp_decay(lr_init, lr_fin, steps)
modelRNN = RNN_model()
K.set_value(modelRNN.optimizer.lr, lr_init)
K.set_value(modelRNN.optimizer.decay, lr_decay)

del embedding_matrix1
gc.collect()

for i in range(3):
    history = modelRNN.fit(X_train, y_train_f
                    , epochs=epochs
                    , batch_size= (BATCH_SIZE+(BATCH_SIZE*(i)))
                    , validation_data = (X_valid, y_valid_f)
                    , verbose=1
                    )
    # Evaluate RMSLE 
    v_rmse = eval_model(modelRNN)
    
print("Finished Fitting the model")
del X_train, y_train_f, X_valid, y_valid_f
gc.collect()
gc.collect()
import time
t1 = time.time()
def load_test():
    for df in pd.read_csv('../input/avito-demand-prediction/test.csv', chunksize= 250000):
        yield df

item_ids = np.array([], dtype=np.int32)
preds= np.array([], dtype=np.float32)

i = 0 
    
for df in load_test():
    
    i +=1
    print(df.dtypes)
    item_id = df['item_id']
    print(" Chunk number is "+str(i))
    test = preprocess_dataset(df)
    test = keras_test_transform(df)
    del df
    gc.collect()
    
    X_test = get_keras_data(test)
    del test 
    gc.collect()
    
    preds1 = modelRNN.predict(X_test, batch_size = BATCH_SIZE, verbose = 1)
    print(preds1.shape)
    del X_test
    gc.collect()
    print("RNN Prediction is done")

    preds1 = preds1.reshape(-1,1)
    #print(predsl.shape)
    preds1 = np.clip(preds1, 0, 1)
    print(preds1.shape)
    item_ids = np.append(item_ids, item_id)
    print(item_ids.shape)
    preds = np.append(preds, preds1)
    print(preds.shape)
    
print("All chunks done")
t2 = time.time()
print("Total time for Parallel Batch Prediction is "+str(t2-t1))
submission = pd.DataFrame( columns = ['item_id', 'deal_probability'])
submission['item_id'] = item_ids
submission['deal_probability'] = preds

print("Check Submission NOW!!!!!!!!@")
submission.to_csv("Avito_Shanth_RNN.csv", index=False)
