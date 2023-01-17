!pip install tensorflow_addons


import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import StratifiedKFold

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from tensorflow.keras.layers import  Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,Average

import os

import warnings

import plotly.graph_objects as go

from textblob import TextBlob

warnings.filterwarnings("ignore")

import plotly.offline as py

py.init_notebook_mode(connected=True)

import tensorflow_addons as tfa

from sklearn.utils import shuffle


def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])



def build_model(transformer, max_len=220):

   

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    seq_out = transformer(input_word_ids)[0]

    pool= GlobalAveragePooling1D()(seq_out)

    



    dense=[]

    FC = Dense(32,activation='relu')

    for p in np.linspace(0.2,0.5,3):

        x=Dropout(p)(pool)

        x=FC(x)

        x=Dense(1,activation='sigmoid')(x)

        dense.append(x)

    

    out = Average()(dense)

    

    model = Model(inputs=input_word_ids, outputs=out)

    optimizer=tfa.optimizers.RectifiedAdam(learning_rate=2e-5,min_lr=1e-6,total_steps=6000)

    model.compile(optimizer, loss=focal_loss(), metrics=[tf.keras.metrics.AUC()])

    

    return model

def build_model(transformer, max_len=220):

   

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    seq_out = transformer(input_word_ids)[0]

    cls_out = seq_out[:,0,:]

    out=Dense(1,activation='sigmoid')(cls_out)

    

    model = Model(inputs=input_word_ids, outputs=out)

    optimizer=tfa.optimizers.RectifiedAdam(learning_rate=2e-5,min_lr=1e-6,total_steps=6000)

    model.compile(optimizer, loss=focal_loss(), metrics=[tf.keras.metrics.AUC()])

    

    return model

    

    

    
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



GCS_DS_PATH = KaggleDatasets().get_gcs_path("jigsaw-multilingual-toxic-comment-classification")

MY_GCS_DS_PATH = KaggleDatasets().get_gcs_path("jigsaw-train-multilingual-coments-google-api")



# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

MODEL = '../input/jplu-tf-xlm-roberta-large'

# First load the real tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)



train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

# Combine train1 with a subset of train2

train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=90000, random_state=0)

])

labels = valid.toxic.unique()

values = valid.toxic.value_counts()



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(title="Target Class distribution")

fig.show()


toxic=train1[train1.toxic==1].id.unique()

non_toxic=train1[train1.toxic==0].id.unique()
cols=['comment_text','lang','toxic','id']

i=0

trans=pd.DataFrame()

path="../input/jigsaw-train-multilingual-coments-google-api/"

for file in os.listdir(path):

    if file.endswith("cleaned.csv"):

        

        df=pd.read_csv(path+file,usecols=['comment_text','toxic','id'])

        df['lang']=file.split("-")[-2]

        

        trans=trans.append([df[df.id.isin(toxic[i*(len(toxic)//6):(i+1)*(len(toxic)//6)])][cols],

                          df[df.id.isin(non_toxic[i*len(non_toxic)//6:(i+1)*len(non_toxic)//6])][cols]])

        i+=1

        

valid= pd.concat([valid[['comment_text','lang','toxic']],trans])

valid = shuffle(valid).reset_index(drop=True)



valid= pd.concat([valid[['comment_text','lang','toxic']],trans])


labels = valid.lang.unique()

values = valid.lang.value_counts()

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_layout(title="Updated validation set language distribution")

fig.show()
import gc

del trans,df,train1,train2

gc.collect()
%%time 



#x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)



#y_train = train.toxic.values

y_valid=valid.toxic.values







test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
def get_fold_data(train_ind,valid_ind):

    

    print("Getting fold data...")

    

    train_x=np.vstack([x_train,x_valid[train_ind]])

    train_y=np.hstack([y_train,y_valid[train_ind]])

    

    valid_x= x_valid[valid_ind]

    valid_y =y_valid[valid_ind]

    

    

    

    train_data  = (

    tf.data.Dataset

    .from_tensor_slices((train_x, train_y))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))

    

    valid_data= (

    tf.data.Dataset

    .from_tensor_slices((valid_x, valid_y))

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)

    return train_data,valid_data
from tensorflow.keras import backend as K



def focal_loss(gamma=1.5, alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed





pred_test=np.zeros((test.shape[0],1))

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)

val_score=[]





for fold,(train_ind,valid_ind) in enumerate(skf.split(x_valid,valid.lang.values)):

    

    if fold < 0:

    

        print("fold",fold+1)

        

        K.clear_session()

        tf.tpu.experimental.initialize_tpu_system(tpu)

        

        train_data,valid_data=get_fold_data(train_ind,valid_ind)

    

        Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"roberta_base_{fold+1}.h5", monitor='val_loss', verbose=1, save_best_only=True,

        save_weights_only=True, mode='min')

        

        with strategy.scope():

            transformer_layer = TFAutoModel.from_pretrained(MODEL)

            model = build_model(transformer_layer, max_len=MAX_LEN)

            

        

        





        n_steps=(x_train.shape[0]+len(train_ind))//BATCH_SIZE

        print(n_steps)



        print("training model {} ".format(fold+1))



        train_history = model.fit(

        train_data,

        steps_per_epoch=n_steps,

        validation_data=valid_data,

        epochs=EPOCHS,callbacks=[Checkpoint],verbose=1)

        

        print("Loading model...")

        model.load_weights(f"roberta_base_{fold+1}.h5")

        

        



        print("fold {} validation auc {}".format(fold+1,np.mean(train_history.history['val_auc'])))

        print("fold {} validation auc {}".format(fold+1,np.mean(train_history.history['val_loss'])))



        val_score.append(np.mean(train_history.history['val_auc']))

        

        print('predict on test....')

        preds=model.predict(test_dataset,verbose=1)



        pred_test+=preds/3

        

#print("Mean cross-validation AUC",np.mean(val_score))



sub['toxic'] = pred_test

sub.to_csv('submission.csv', index=False)

sub.head()