# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")

print("Shape of Train Data ",train.shape)

test=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")

print("Shape of Test Data ",test.shape)
train.head()
!pip install nlp

import nlp
snli= nlp.load_dataset("snli")
snli
print("Type of SNLI Dataset ",type(snli))
print(snli['train'])

print("Type of SNLI Train ",type(snli['train']))
snli_train_df = pd.DataFrame(snli['train'])

snli_train_df.head()
test_dup=pd.merge(test[['premise','hypothesis']],snli_train_df[['premise','hypothesis']],how="outer",indicator=True)

test_dup.head()
test_dup['_merge'].unique()

snli_train_df['lang_abv']="en"

train_df=train[['premise','hypothesis','lang_abv','label']]

snli_train_df['dataset']="snli"

train_df['dataset']="train"

train_df=pd.concat([train_df,snli_train_df])

print("Shape of Original Train Data ",train.shape)

print("Shape of SNLI Train Data ",snli_train_df.shape )

print("Shape after merging Original Train Data With SNLI ",train_df.shape)
train_df.head()
xnli=nlp.load_dataset("xnli")

xnli
for idx, elt in enumerate(xnli['validation']):

    

    print('premise:', elt['premise'])

    print('hypothesis:', elt['hypothesis'])

    print('label:', elt['label'])

    print('label name:', xnli['validation'].features['label'].names[elt['label']])

    print('-' * 80)

    

    if idx >= 3:

        break
buffer = {

    'premise': [],

    'hypothesis': [],

    'label': [],

    'lang_abv': []

}







for x in xnli['validation']:

    label = x['label']

    for idx, lang in enumerate(x['hypothesis']['language']):

        hypothesis = x['hypothesis']['translation'][idx]

        premise = x['premise'][lang]

        buffer['premise'].append(premise)

        buffer['hypothesis'].append(hypothesis)

        buffer['label'].append(label)

        buffer['lang_abv'].append(lang)

        

# convert to a dataframe and view

xnli_valid_df = pd.DataFrame(buffer)

xnli_valid_df = xnli_valid_df[['premise', 'hypothesis', 'label', 'lang_abv']]
xnli_valid_df.shape
test_dup=pd.merge(test[['premise','hypothesis']],xnli_valid_df[['premise','hypothesis']],how="outer",indicator=True)

test_dup.head()
test_dup['_merge'].value_counts()
dup_pairs=test_dup[test_dup['_merge']=="both"]

dup_pairs.head()
dup_pairs['combo']=dup_pairs['premise']+" "+dup_pairs['hypothesis']

dup_pairs.head()
xnli_valid_df['combo']=xnli_valid_df['premise']+" "+xnli_valid_df['hypothesis']
xnli_valid_without_dups=xnli_valid_df[(~xnli_valid_df['combo'].isin(dup_pairs['combo'].tolist()))]
xnli_valid_without_dups.drop(['combo'],axis=1,inplace=True)
xnli_valid_without_dups.shape
xnli_valid_without_dups['dataset']="xnli"
train_df=pd.concat([train_df,xnli_valid_without_dups])

train_df.shape
mnli=nlp.load_dataset(path='glue', name='mnli')

mnli
mnli_train=pd.DataFrame(mnli['train'])

mnli_train.head()
mnli_train.shape
test_dup=pd.merge(test[['premise','hypothesis']],mnli_train[['premise','hypothesis']],how="outer",indicator=True)

test_dup['_merge'].unique()
mnli_train.drop(['idx'],axis=1,inplace=True)
mnli_train['lang_abv']="en"

mnli_train['dataset']="mnli"
train_df=pd.concat([train_df,mnli_train])

train_df.shape
pd.isnull(train_df).sum()
from transformers import BertTokenizer,TFBertModel

import tensorflow as tf

from transformers import AutoTokenizer

MODEL_NAME="jplu/tf-xlm-roberta-large"

MAX_LEN=64

BATCH_SIZE=64
tokeniser=AutoTokenizer.from_pretrained(MODEL_NAME) ## Autokeniser will initialise the tokeniser based on the model name

from tensorflow.keras import Input, Model, Sequential

from tensorflow.keras.layers import Dense, Dropout

from keras.optimizers import Adam

from transformers import TFAutoModel
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print("In TPU STRATERGY")

    print('Number of replicas:', strategy.num_replicas_in_sync)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
def createModel():

    with strategy.scope():

        model=TFAutoModel.from_pretrained(MODEL_NAME)

        input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token', dtype='int32')

        #input_mask=tf.keras.layers.Input(shape=(MAX_LEN,), name='input_mask', dtype='int32')

        #input_token_ids=tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token_ids', dtype='int32')

        ### From the Model, we need to extract the Last Hidden Layer - this is the first element of the model output

        embedding=model(input_ids)[0]

        ### Extract the CLS Token from the Embedding Layer. CLS Token is aggregate of the entire sequence representation. It is the first token

        cls_token=embedding[:,0,:] ## embedding is of the size batch_size*MAX_LEN*768

    

        ### Add a Dense Layer, with three outputs 

        output_layer = Dense(3, activation='softmax')(cls_token)

    

        classification_model= Model(inputs=input_ids, outputs = output_layer)

    

        classification_model.compile(Adam(lr=1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    

        #classification_model.summary()

    

    return classification_model

    

    

    

    
'''

This function will take the complete data, with option to select certain external datasets. It will also split the data into train and validation Split,

encode the data and create a TF.Data.DataSet object for the train and validation data

'''

def createTFDataSet(data,external_dataset=None,test_size=0.2,padding=True,max_length=MAX_LEN,truncation=True,batch_size=BATCH_SIZE):

    tokeniser=AutoTokenizer.from_pretrained(MODEL_NAME) 

    if external_dataset==None:

        train_data=data[data['dataset']=="train"]

    else:

        dat=data[data['dataset']=="train"]

        external_data=data[data['dataset'].isin(external_dataset)]

        train_data=pd.concat([dat,external_data])

        assert dat.shape[0]+external_data.shape[0]==train_data.shape[0]

    ### Split the Data into Train and Validation Split

    X=train_data[['hypothesis','premise']]

    y=train_data['label']

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Shape of Train Data ",X_train.shape)

    print("Shape of Test Data ",X_test.shape)

    

    ### Encode the Training and Validation Data

    train_encoded=tokeniser.batch_encode_plus(X_train[['hypothesis','premise']].values.tolist(),pad_to_max_length=padding,max_length=max_length,truncation=True)

    val_encoded=tokeniser.batch_encode_plus(X_test[['hypothesis','premise']].values.tolist(),pad_to_max_length=padding,max_length=max_length,truncation=True)

    

    ### Convert the Encoded Train and Validation data into TF Dataset

    auto = tf.data.experimental.AUTOTUNE

    train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_encoded['input_ids'], y_train))

    .repeat()

    .shuffle(2048)

    .batch(batch_size)

    .prefetch(auto))

    

    

    valid_dataset = (tf.data.Dataset

    .from_tensor_slices((val_encoded['input_ids'], y_test))

    .batch(batch_size)

    .cache()

    .prefetch(auto))

    

    return tokeniser,train_dataset,valid_dataset,X_train.shape[0]

    

    

    
tokeniser,train_dataset,valid_dataset,train_rows=createTFDataSet(train_df,external_dataset=['xnli','mnli'])
with strategy.scope():

    model=createModel()

    model.summary()
n_steps =  train_rows//BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=5

)
test_encoded=tokeniser.batch_encode_plus(test[['hypothesis','premise']].values.tolist(),pad_to_max_length=True,max_length=MAX_LEN,truncation=True)





test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_encoded['input_ids'])

    .batch(BATCH_SIZE)

)

test_preds = model.predict(test_dataset, verbose=1)

predictions = test_preds.argmax(axis=1)
submission = pd.DataFrame()

submission['id']=test['id'].tolist()

submission['prediction'] = predictions
submission.to_csv("submission.csv",index=False)
model.save_weights("XLM_R_MNLI_XNLI.h5",overwrite=True)


        