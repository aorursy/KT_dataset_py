import pandas as pd 
import numpy as np 
import time 
import gc 

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model

from keras import optimizers

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
    print("Casting data types to type Category.......")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
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
    gc.collect()
    
    print("Loading Test for Label Encoding on Train + Test")
    use_cols_test = ['region', 'city', 'parent_category_name', 'category_name', 'title', 'description', 'price']
    test = pd.read_csv("../input/avito-demand-prediction/test.csv", usecols = use_cols_test)

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

    train = DF[0:ntrain]
    del DF 
    gc.collect()
    
    train['price'] = np.log1p(train['price'])
    
    return train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name

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
    
    del dataset['title_description']
    gc.collect()

    dataset['region'] = le_region.transform(dataset['region'])
    dataset['city'] = le_city.transform(dataset['city'])
    dataset['category_name'] = le_category_name.transform(dataset['category_name'])
    dataset['parent_category_name'] = le_parent_category_name.transform(dataset['parent_category_name'])
    dataset['price'] = np.log1p(dataset['price'])
    
    return dataset
    
def get_keras_data(dataset):
    X = {
        'seq_title_description': pad_sequences(dataset.seq_title_description, maxlen=max_seq_title_description_length)
        ,'region': np.array(dataset.region)
        ,'city': np.array(dataset.city)
        ,'category_name': np.array(dataset.category_name)
        ,'parent_category_name': np.array(dataset.parent_category_name)
        ,'price': np.array(dataset[["price"]])

    }
    
    print("Data ready for Vectorization")
    
    return X

# Loading Train data - No Params, No Image data 
dtypes_train = {
                'price': 'float32',
                'deal probability': 'float32',
}

# No user_id
use_cols = ['region', 'city', 'parent_category_name', 'category_name', 'title', 'description', 'price','deal_probability']
train = pd.read_csv("../input/avito-demand-prediction/train.csv", usecols = use_cols, dtype = dtypes_train, nrows = 10000)

y_train = np.array(train['deal_probability'])
del train['deal_probability']
gc.collect()

max_seq_title_description_length = 100
max_words_title_description = 200000

train = preprocess_dataset(train)
train, tokenizer, le_region, le_city, le_category_name, le_parent_category_name = keras_fit(train)
train = keras_train_transform(train)
print("Tokenization done and TRAIN READY FOR Validation splitting")

# Calculation of max values for Categorical fields 

max_region = np.max(train.region.max())+2
max_city= np.max(train.city.max())+2
max_category_name = np.max(train.category_name.max())+2
max_parent_category_name = np.max(train.parent_category_name.max())+2

EMBEDDING_DIM1 = 300
EMBEDDING_FILE1 = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

vocab_size = len(tokenizer.word_index)+2
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
    seq_title_description = Input(shape=[100], name="seq_title_description")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    price = Input(shape=[1], name="price")
    
    #Embeddings layers

    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights = [embedding_matrix1], trainable = False)(seq_title_description)
    emb_region = Embedding(vocab_size, 10)(region)
    emb_city = Embedding(vocab_size, 10)(city)
    emb_category_name = Embedding(vocab_size, 10)(category_name)
    emb_parent_category_name = Embedding(vocab_size, 10)(parent_category_name)
    
    rnn_layer1 = GRU(50) (emb_seq_title_description)
    
    #main layer
    main_l = concatenate([
          rnn_layer1
        , Flatten() (emb_region)
        , Flatten() (emb_city)
        , Flatten() (emb_category_name)
        , Flatten() (emb_parent_category_name)
        , price
    ])
    
    main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="sigmoid") (main_l)
    
    #model
    model = Model([seq_title_description, region, city, category_name, parent_category_name, price], output)
    model.compile(optimizer = 'adam',
                  loss= root_mean_squared_error,
                  metrics = [root_mean_squared_error])
    return model

def rmse(y, y_pred):

    Rsum = np.sum((y - y_pred)**2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum/n)
    return RMSE 

def eval_model(model, X_test1):
    val_preds = model.predict(X_test1)
    y_pred = val_preds[:, 0]
    
    y_true = np.array(y_test1)
    
    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)
    
    print(yt.isnull().any())
    print(yp.isnull().any())
    
    v_rmse = rmse(y_true, y_pred)
    print(" RMSE for VALIDATION SET: "+str(v_rmse))
    return v_rmse

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
def predictions(model):
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
        print(test.dtypes)

        test = keras_test_transform(test)
        del df
        gc.collect()
    
        print(test.dtypes)
    
        X_test = get_keras_data(test)
        del test 
        gc.collect()
    
        Batch_Size = 512*3
        preds1 = modelRNN.predict(X_test, batch_size = Batch_Size, verbose = 1)
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
    return preds 
del train['description']
del train['title']
train.dtypes
# Converting the TRAIN Data frame into array values 
train1 = np.array(train.values)
del train
gc.collect()

# Function to Convert the Train array values back into a data frame 
# The data frame produced by this function will be used as inputs for the RNN 

def get_data_frame(dataset):
    
    DF = pd.DataFrame()
    
    DF['category_name'] = np.array(dataset[:,0])
    DF['city'] = np.array(dataset[:,1])
    DF['parent_category_name'] = np.array(dataset[:,2])
    DF['price'] = np.array(dataset[:,3])
    DF['region'] = np.array(dataset[:,4])
    DF['seq_title_description'] = np.array(dataset[:,5])
    
    return DF 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time 
skf = KFold(n_splits = 3)
Kfold_preds_final = []
k = 0
RMSE = []

for train_idx, test_idx in skf.split(train1, y_train):
    
    print("Number of Folds.."+str(k+1))
    
    # Initialize a new Model for Current FOLD 
    
    epochs = 1
    batch_size = 512 * 3
    steps = (int(train1.shape[0]/batch_size))*epochs
    lr_init, lr_fin = 0.002, 0.001
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)

    #K Fold Split 
    
    X_train1, X_test1 = train1[train_idx], train1[test_idx]
    print(X_train1.shape, X_test1.shape)
    y_train1, y_test1 = y_train[train_idx], y_train[test_idx]
    print(y_train1.shape, y_test1.shape)
    gc.collect()
    
    # Getting the dataframes for Training and Test Arrays 
    X_train_final = get_data_frame(X_train1)
    X_test_final = get_data_frame(X_test1)
    
    del X_train1, X_test1
    gc.collect()
    
    # Getting the Dictionary for RNN input 
    X_train_f = get_keras_data(X_train_final)
    X_test_f = get_keras_data(X_test_final)
    
    del X_train_final, X_test_final
    gc.collect()

    # Fit the NN Model 
    for i in range(3):
        hist = modelRNN.fit(X_train_f, y_train1, batch_size=batch_size+(batch_size*(2*i)), epochs=epochs, validation_data=(X_test_f, y_test1), verbose=1)

    del X_train_f
    gc.collect()

    # Print RMSE for Validation set for Kth Fold 
    v_rmse = eval_model(modelRNN, X_test_f)
    RMSE.append(v_rmse)
    
    del X_test_f
    del y_train1, y_test1
    gc.collect()
    
    # Predict test set for Kth Fold 
    preds = predictions(modelRNN)
    del modelRNN 
    gc.collect()

    print("Predictions done for Fold "+str(k))
    print(preds.shape)
    Kfold_preds_final.append(preds)
    del preds
    gc.collect()
    print("Number of folds completed...."+str(len(Kfold_preds_final)))
    print(Kfold_preds_final[k][0:10])

print("All Folds completed"+str(k+1))   
print("RNN FOLD MODEL Done")
pred_final1 = np.average(Kfold_preds_final, axis =0) # Average of all K Folds
print(pred_final1.shape)

# Find lowest RMSE value 
min_value = min(RMSE)
# Which KFOLD has this lowest RMSE value 
RMSE_idx = RMSE.index(min_value)
print(RMSE_idx)

# What is the prediction values corresponding to the lowest RMSE value 
pred_final2 = Kfold_preds_final[RMSE_idx]
print(pred_final2.shape)

del Kfold_preds_final, train1
gc.collect()
# Average Output 
pred_final1[0:5]
# Output with lowest RMSE 
pred_final2[0:5]
test_cols = ['item_id']
test = pd.read_csv('../input/avito-demand-prediction/test.csv', usecols = test_cols)

# using Average of KFOLD preds 

submission1 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission1['item_id'] = test['item_id']
submission1['deal_probability'] = pred_final1

print("Check Submission NOW!!!!!!!!@")
submission1.to_csv("Avito_Shanth_RNN_AVERAGE.csv", index=False)

# Using KFOLD preds with Minimum value 
submission2 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

submission2['item_id'] = test['item_id']
submission2['deal_probability'] = pred_final2

print("Check Submission NOW!!!!!!!!@")
submission2.to_csv("Avito_Shanth_RNN_MIN.csv", index=False)
