import numpy as np 
import os
import gc
import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Sklearn model definition 

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

# Keras model definition 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, GlobalAveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras.layers import Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

# Tensor Flow 

import tensorflow as tf

start_time = time.time()
# RNN Preprocessing, Transformation methods 

def preprocess_RNN(dataset):
    
    print("Filling Missing Values")
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
    
    print("Casting data types to type Category")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')  
    print("RNN PreProcessing completed")
    
    return dataset 

def fit_RNN_text(dataset):

    print("Fit Name and Item description fields for Tokenization")
    raw_text = np.hstack([dataset.item_description.str.lower(), dataset.name.str.lower()])
    word_token = Tokenizer()
    word_token.fit_on_texts(raw_text)
    print("RNN Data fit completed")
    
    return word_token

def fit_RNN_label(dataset):

    print("Fit Categorical variables on full Merged Test and Train data")
    
    le_name = LabelEncoder()
    le_name.fit(dataset.category_name)
    le_brand = LabelEncoder()
    le_brand.fit(dataset.brand_name)
    
    print("Completed Label fitting")
    return le_name, le_brand

def transform_RNN(dataset, le_name, le_brand, word_token):
    print("Use Defined Label encoders to Encode brand and category_name")
    dataset['category'] = le_name.transform(dataset.category_name)
    dataset['brand'] = le_brand.transform(dataset.brand_name)
    print("Convert Text to sequences")
    dataset["seq_item_description"] = word_token.texts_to_sequences(dataset.item_description.str.lower())
    dataset["seq_name"] = word_token.texts_to_sequences(dataset.name.str.lower())
    print("Sequence Conversion Completed")
    
    return dataset 

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    
    print("Data ready for Vectorization")
    
    return X
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
def RNN_model():

    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 60
    
    emb_name = Embedding(MAX_TEXT, emb_size//3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    # Try Global Average Max Pooling 
    
    #emb_name = GlobalAveragePooling1D()(emb_name)
    #emb_item_desc = GlobalAveragePooling1D()(emb_item_desc)
    
    # RNN For item desc and name 
    rnn_layer1 = GRU(25) (emb_item_desc)
    #rnn_layer1 = Attention(X_train['item_desc'].shape[1]-1)(rnn_layer1)
    rnn_layer2 = GRU(12) (emb_name)
    #rnn_layer2 = Attention(X_train['name'].shape[1]-1)(rnn_layer2)
    
    #main layer
    main_l = concatenate([
         Flatten() (emb_brand)
        , Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.1)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,activation='relu') (main_l))
 
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([brand, category, item_condition, item_desc, name, num_vars], output)
    optimizer = optimizers.Adam()
    model.compile(loss="mse", 
                  optimizer=optimizer)
    return model

def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)
    y_pred = val_preds[:, 0]
    
    y_true = np.array(valid_prices)
    
    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)
    
    print(yt.isnull().any())
    print(yp.isnull().any())
    
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
# Loading data 
t1 = time.time()

train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')
train = train[train['price'] !=0 ]
ntrain = train.shape[0]
train['target'] = np.log1p(train['price'])
test = pd.read_table('../input/mercari-price-suggestion-challenge/test_stg2.tsv', sep='\t')
ntest = test.shape[0]
merge = pd.concat([train, test], 0, ignore_index = True)

del test, train
gc.collect()
# Training RNN 
# Pre Process RNN Data - Missing data, Tokenization 

t1 = time.time()
merge = preprocess_RNN(merge)
t2 = time.time()
print("Time taken for RNN preprocess "+str(t2-t1))
print(merge.shape)

# Fitting Categorical columns on test and train data 
t1 = time.time()
le_name, le_brand = fit_RNN_label(merge)
t2 = time.time()
print("Time taken to Fit on RNN "+str(t2-t1))

# Fitting Text data on Train data only 
t1 = time.time()
train = merge[:ntrain]
word_token = fit_RNN_text(train)

# Transforming training data to be readied for RNN TRAINING 
train = transform_RNN(train, le_name, le_brand, word_token)
t2 = time.time()
print("Time taken to Fit and Transform on RNN "+str(t2-t1))
print(train.shape, merge.shape)
del merge
gc.collect
#EMBEDDINGS MAX VALUE
#Max_text is USED TO CALCULATE VOCABULARY LENGTH for EMBEDDING 
#MAX_Seq values are USED TO CALCULATE PADDING LENGTHS 
# For padding 
MAX_NAME_SEQ = 20 
MAX_ITEM_DESC_SEQ = 60 
MAX_CATEGORY_NAME_SEQ = 20 
MAX_CATEGORY = np.max(train.category.max())+1
MAX_BRAND = np.max(train.brand.max())+1
MAX_CONDITION = 6
# For Vocab length --> THE VALUE OF MAX_TEXT is THE VALUE OF THE VOCABULARY LENGTH 
MAX_TEXT = np.max([np.max(train.seq_name.max()) , np.max(train.seq_item_description.max())])+2
# Split into test and train data 

dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.99)
dtrain['target'] =np.log1p(dtrain['price'])
target  = np.array(dtrain.target)
dvalid['target'] =np.log1p(dvalid['price'])
valid_prices = np.array(dvalid.target)
print(dvalid.shape, dtrain.shape)
#Sequence padding 
del train
X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
del dtrain, dvalid
gc.collect()
#FITTING THE MODEL
epochs = 1
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.0045
lr_decay = exp_decay(lr_init, lr_fin, steps)
modelRNN = RNN_model()
K.set_value(modelRNN.optimizer.lr, lr_init)
K.set_value(modelRNN.optimizer.decay, lr_decay)
# Fitting RNN 
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
for i in range(3):
    history = modelRNN.fit(X_train, target
                    , epochs=epochs
                    , batch_size=BATCH_SIZE+(512*i)
                    , validation_data = (X_valid, valid_prices)
                    , verbose=1
                    )
    # Evaluate RMSLE 
    v_rmsle = eval_model(modelRNN)
    print('[{}] Finished predicting valid set...'.format(time.time() - start_time))
    
print("Finished Fitting the model")
del X_train, X_valid
del target, valid_prices
del epochs, lr_init, lr_fin, exp_decay, lr_decay
gc.collect()
t1 = time.time()
train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')
train = train[train['price'] != 0]
train['target'] = np.log1p(train['price'])
n_train = train.shape[0]
def split_cat(text):
    try: 
        s1, s2, s3 = text.split("/")
        return s1, s2, s3
    
    except: return ("No Label", "No Label", "No Label")

def wordbatch_preprocess(df):
    df["category_name"] = df["category_name"].fillna(value="missing").astype(str)
    df["name"] = df["name"].fillna(value="missing").astype(str)
    df["brand_name"] = df["brand_name"].fillna(value="missing").astype(str)
    df["item_description"] = df["item_description"].fillna(value="missing").astype(str)
    df["item_condition_id"] = df["item_condition_id"].astype(int)
    df["shipping"] = df["shipping"].astype(int)
    print("Got so far")
    
    print(df.dtypes)
    df['subcat_0'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))
    print(df.dtypes)
    
    return df

# Pre Processing for word batch 
train = wordbatch_preprocess(train)

print("Pre Process for Word batch is done")
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

class WordBatchModel(object):
    def __init__(self):
        self.wb_desc = None
        self.desc_indices = None
        self.cv_name, self.cv_name2 = None, None
        self.cv_cat0, self.cv_cat1, self.cv_cat2 = None, None, None
        self.cv_brand = None
        self.cv_condition = None
        self.cv_cat_brand = None
        self.desc3 = None
        self.model = None

    def train_wbm(self, df):

        self.wb_desc = wordbatch.WordBatch(None,
                                           extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                "idf": None}), procs=8)
        self.wb_desc.dictionary_freeze = True
        X_desc = self.wb_desc.fit_transform(df['item_description'])
        self.desc_indices = np.array(np.clip(X_desc.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        X_desc = X_desc[:, self.desc_indices]
        
        self.cv_cat0 = CountVectorizer(min_df=2)
        X_category0 = self.cv_cat0.fit_transform(df['subcat_0'])
        
        self.cv_cat1 = CountVectorizer(min_df=2)
        X_category1 = self.cv_cat1.fit_transform(df['subcat_1'])
        
        self.cv_cat2 = CountVectorizer(min_df=2)
        X_category2 = self.cv_cat2.fit_transform(df['subcat_2'])

        self.cv_brand = CountVectorizer(min_df=2, token_pattern=".+")
        X_brand = self.cv_brand.fit_transform(df['brand_name'])

        # Variations 

        self.cv_name = CountVectorizer(min_df=2, ngram_range=(1, 1),binary=True, token_pattern="\w+")
        X_name = 2 * self.cv_name.fit_transform(df['name'])
        
        self.cv_name2 = CountVectorizer(min_df=2, ngram_range=(2, 2), binary=True, token_pattern="\w+")
        X_name2 = 0.5 * self.cv_name2.fit_transform(df['name'])
                                                      
        df["cat_brand"] = [a + " " + b for a, b in zip(df["category_name"], df["brand_name"])]
        self.cv_cat_brand = CountVectorizer(min_df=10, token_pattern=".+")
        X_cat_brand = self.cv_cat_brand.fit_transform(df["cat_brand"])
        
        self.cv_condition = CountVectorizer(token_pattern=".+")
        X_condition = self.cv_condition.fit_transform((df['item_condition_id'] + 10 * df["shipping"]).apply(str))
            
        self.desc3 = CountVectorizer(ngram_range=(3, 3), max_features=1000, binary=True, token_pattern="\w+")
        X_desc3 = self.desc3.fit_transform(df["item_description"])

        X = hstack((X_condition,
                    X_desc, X_brand,
                    X_category0, X_category1, X_category2,
                    X_name, X_name2,
                    X_cat_brand, 
                    X_desc3
                    )).tocsr()

        print("X Reconstructed")
        y = df['target'].values
        #y = y.reshape(y.shape[0],1)
        print("Y created")

        self.model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X.shape[1], alpha_fm=0.02, L2_fm=0.0,
                             init_fm=0.01, D_fm=200, e_noise=0.0001, iters=10, inv_link="identity", threads=4)
        print("Model Defined")
        self.model.fit(X, y)

    def predict(self, df):
        X_desc = self.wb_desc.transform(df["item_description"])
        X_desc = X_desc[:, self.desc_indices]
        
        X_brand = self.cv_brand.transform(df['brand_name'])

        X_name = 2 * self.cv_name.transform(df["name"])
        X_name2 = 0.5 * self.cv_name2.transform(df["name"])

        X_category0 = self.cv_cat0.transform(df['subcat_0'])
        X_category1 = self.cv_cat1.transform(df['subcat_1'])
        X_category2 = self.cv_cat2.transform(df['subcat_2'])
        
        X_condition = self.cv_condition.transform((df['item_condition_id'] + 10 * df["shipping"]).apply(str))
        
        df["cat_brand"] = [a + " " + b for a, b in zip(df["category_name"], df["brand_name"])]
        X_cat_brand = self.cv_cat_brand.transform(df["cat_brand"])
        
        X_desc3 = self.desc3.transform(df["item_description"])

        X = hstack((X_condition,
                    X_desc, X_brand,
                    X_category0, X_category1, X_category2,
                    X_name, X_name2,
                    X_cat_brand, 
                    X_desc3
                   )).tocsr()

        return self.model.predict(X)
# Training WordBatch 
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
print(train.shape)
t1 = time.time()
wbm = WordBatchModel()
wbm.train_wbm(train)
t2 = time.time()
print("Time taken for Word Batch is "+str(t2-t1))
del train 
gc.collect()
import time
t1 = time.time()
def load_test():
    for df in pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', sep='\t', chunksize= 1000000):
        yield df

test_ids = np.array([], dtype=np.int32)
preds= np.array([], dtype=np.float32)

i = 0 
    
for df in load_test():
    
    i +=1
    print(" Chunk number is "+str(i))
    df1 = df2 = df
    testRNN = preprocess_RNN(df1)
    print(df1.dtypes)
    print(testRNN.dtypes)
    testRNN = transform_RNN(testRNN, le_name, le_brand, word_token)
    print(testRNN.dtypes)
    X_testRNN = get_keras_data(testRNN)
    predsRNN1 = modelRNN.predict(X_testRNN, batch_size = BATCH_SIZE, verbose = 1)
    test_id = df['test_id']
    del testRNN
    del df['seq_item_description'], df['seq_name'], df['brand'], df['category']
    gc.collect()
    print("RNN Prediction is done")
    
    print(df.isnull().any())
    print(df.dtypes)
    test_WB = wordbatch_preprocess(df2)
    print(df.dtypes)
    print("Word batch preprocess done")
    predsWB1 = wbm.predict(test_WB)
    print("Word Batch Prediction done")

    predsRNN = np.expm1(predsRNN1)
    predsWB = np.expm1(predsWB1)
    
    predsRNN = predsRNN.reshape(predsRNN.shape[0],1)
    predsWB = predsWB.reshape(predsWB.shape[0],1)

    predsRNN = np.clip(predsRNN, 0, predsRNN.max())
    predsWB = np.clip(predsWB, 0, predsWB.max())
    
    preds= np.append(preds, ((predsRNN*0.4) + (predsWB*0.6)))
    test_ids = np.append(test_ids, test_id)
    
print("All chunks done")
t2 = time.time()
print("Total time for Parallel Batch Prediction is "+str(t2-t1))
import pandas as pd 

submission = pd.DataFrame( columns = ['test_id', 'price'])
submission['test_id'] = test_ids
submission['price'] = preds

print("Check Submission NOW!!!!!!!!@")
submission.to_csv("BatchPrediction_MemoryCoreOptimized.csv", index=False)
submission
t2 = time.time()
print("Total time taken is "+str(t2-start_time))