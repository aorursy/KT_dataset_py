# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

train = pd.read_csv("../input/chinese-text-multi-classification/nCoV_100k_train.labled.csv")

test = pd.read_csv("../input/chinese-text-multi-classification/nCov_10k_test.csv")

train.head(2)
labled_train=pd.read_csv("/kaggle/input/unlabled-train-data-sample/train_unlable_sample.csv")

#labled_train=pd.read_csv("/kaggle/input/fake-data-8w/train_unlable_sample_8w.csv")

labled_train.drop(['id','date','user','image','vedio'],axis = 1,inplace = True)

#labled_train=labled_train.sample(n=40000,random_state=2020)

labled_train.head(2)
columns=['id','date','user','content','image','vedio','target']

train.columns=columns

test.columns=columns[:-1]

train.drop(['id','date','user','image','vedio'],axis = 1,inplace = True)

train.head()
train['target'].value_counts()
train_1=train.loc[(train['target']=='-1')] 

train0=train.loc[(train['target']=='0')] 

train1=train.loc[(train['target']=='1')] 



train_1.loc[:,'target']=-1

train0.loc[:,'target']=0

train1.loc[:,'target']=1



train = pd.concat([train_1,train0,train1])

print(train['target'].value_counts())

train.head()
train = pd.concat([train,labled_train])

print(train['target'].value_counts())
train.reset_index(drop=True,inplace=True)

train.head()
train=train.sample(frac=1,random_state=2020)

train.head()
EPOCHS=10

MAX_SEQUENCE_LENGTH=220
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import tensorflow_hub as hub

import numpy as np

print(tf.__version__)
from transformers import *
#加载tokenizer和model

#pretrained = "hfl/chinese-roberta-wwm-ext"

pretrained = 'bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(pretrained)

pretrained_model = TFBertModel.from_pretrained(pretrained)
tokenizer.tokenize("学习自然语言处理的BERT模型")
tokenizer.encode_plus("学习自然语言处理的BERT模型",max_length=220)
#train_input = text_encode(train['content'].astype(str), tokenizer, max_len=MAX_SEQUENCE_LENGTH)

#test_input = text_encode(test['content'].astype(str), tokenizer, max_len=MAX_SEQUENCE_LENGTH)



train_input = tokenizer.batch_encode_plus(train['content'].astype(str), max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, return_tensors='tf')

test_input = tokenizer.batch_encode_plus(test['content'].astype(str), max_length=MAX_SEQUENCE_LENGTH, pad_to_max_length=True, return_tensors='tf')



train_labels = train['target'].astype(int)+1
def create_model(pretrained_model):

    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')

    token_type_ids = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='token_type_ids')

    attention_mask = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')

    

    # Use pooled_output(hidden states of [CLS]) as sentence level embedding

    pooled_output = pretrained_model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})[1]

    

    output = Dense(3, activation='sigmoid')(pooled_output)

    model = Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, outputs=output)

    model.compile(Adam(lr=2e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
def create_model_multi_dropout(pretrained_model):

    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')

    token_type_ids = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='token_type_ids')

    attention_mask = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')

    

    # Use pooled_output(hidden states of [CLS]) as sentence level embedding

    pooled_output = pretrained_model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})[1]

    

    # multi dropout

    dropouts = [Dropout(0.1) for _ in range(3)]

    for  i, dropout in enumerate(dropouts) :

        if i==0:

            out = dropout( pooled_output)

            output = Dense(3, activation='sigmoid')(out)

        else:

            temp_out = dropout ( pooled_output )

            output = output + Dense(3, activation='sigmoid')(temp_out)

    output = output / len( dropouts)        



    model = Model(inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, outputs=output)

    #model.compile(Adam(lr=2e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.compile(Adam(lr=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
#model=create_model(pretrained_model)

model = create_model_multi_dropout(pretrained_model)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=16,

    callbacks=[EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]

)
model.save_weights('bert_hf_model_epoch10_addlabeled_data_multi_dropout.h5')
test_pred = model.predict(test_input)

predictions = np.argmax(test_pred, axis=-1)-1

print(predictions)
submission=pd.read_csv("/kaggle/input/chinese-text-multi-classification/submit_example.csv")

submission['y']=predictions

submission.to_csv('submission_bert_hf_addlabeled_data_multi_dropout.csv', index=False)
res=[26.58379066, 26.9272789 , 24.38326198]

test_pred_op=test_pred*res

predictions_op = np.argmax(test_pred_op, axis=-1)-1

submission=pd.read_csv("/kaggle/input/chinese-text-multi-classification/submit_example.csv")

submission['y']=predictions_op

submission.to_csv('submission_bert_hf_addlabeled_data_multi_dropout_optimize.csv', index=False)