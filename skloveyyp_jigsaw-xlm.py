import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# 遍历kaggle目录，检查文件是否载入正确。

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# 加载packages。

import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.keras.layers import Dense, Input,concatenate,Bidirectional, LSTM,MaxPool1D,MaxPool3D,GlobalMaxPooling1D,GlobalAveragePooling1D

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint



import transformers

from transformers import *

from transformers import TFAutoModel, AutoTokenizer,AutoModel

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors





from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from sklearn.metrics import f1_score

import re



def seed_everything(seed=0):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



# 设定seed

SEED = 36

seed_everything(SEED)

#from kaggle_datasets import KaggleDatasets
# 调用bert tokenizer将text转化为tokens（以及mask matrix）

def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=True, 

        return_token_type_ids=True,

        pad_to_max_length=True,

        max_length=maxlen

     )

    #,np.asarray(enc_di['attention_masks'],dtype=np.int32)

    return np.array(enc_di['input_ids'],dtype=np.int32),np.array(enc_di['attention_mask'],dtype=np.int32),np.array(enc_di['token_type_ids'],dtype=np.int32)
from tensorflow.keras import backend as K

# 将loss function换成focal，效果不明显。



def focal_loss(gamma=2., alpha=.2):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
# tf model structure

def build_model(transformer,max_len=512):



    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")#input_word_ids

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    sequence_output = transformer((input_ids,input_mask,segment_ids))[0]

    

    #cls_token = sequence_output[:, 0, :]

    gp = GlobalMaxPooling1D()(sequence_output)

    ap = GlobalAveragePooling1D()(sequence_output)

    stack = concatenate([gp,ap],axis=1)

    #bert的输出使用max和mean，也可以像上一个comment中一样只用CLS，但此比赛CLS效果不佳。

    

    out = Dense(1, activation='sigmoid')(stack)

    #在bert的输出pooling之后下接linear layer。

    model = Model(inputs=[input_ids,input_mask,segment_ids], outputs=out)

    #定义模型的输入与输入。

    model.compile(Adam(lr=0.25e-5), loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()]) 

    #设定学习率Loss function和评分metrics。

    

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



# 设定使用TPU需要的参数。
AUTO = tf.data.experimental.AUTOTUNE



# Data access

#GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# 训练的Configuration

EPOCHS = 2

BATCH_SIZE = 16* strategy.num_replicas_in_sync

MAX_LEN = 256

MODEL = 'jplu/tf-xlm-roberta-large'

# 从hugging face的目录中选择模型型号，这里选用XLMR-large，也可以换成mbert、XLMR-base等。
# First load the real tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

# 载入数据
df_train = pd.read_csv("/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv")

df_train['lang'] = 'lang'

df_train = df_train[['comment_text', 'toxic', 'lang']].head(0)

# 载入数据




for lang in ['es', 'tr', 'it', 'ru', 'pt', 'fr']:

                                                      

    df_lang = pd.read_csv(f'../input/jigsawtanslatedgoogle/jigsaw-unintended-bias-train_{lang}_clean.csv')[['id', 'comment_text', 'toxic']]

    df_lang = df_lang[(df_lang['toxic'] < 0.2) | (df_lang['toxic'] > 0.5)]

    df_lang['toxic'] = df_lang['toxic'].apply(lambda x:round(x)).astype(np.int32)

    df_lang = df_lang[~df_lang['comment_text'].isna()]

    df_lang_sampled = pd.concat([

        df_lang[['comment_text', 'toxic']].query('toxic==1').sample(n=35000),

        df_lang[['comment_text', 'toxic']].query('toxic==0').sample(n=80000),])  

    del df_lang

#     df_lang = df_lang.drop_duplicates(subset='comment_text')                                                                

    df_lang_sampled['lang'] = lang

    df_train = df_train.append(df_lang_sampled) 

# 载入数据
def text_head_tail(input_df, text_col='comment_text'):

    

    df = input_df.copy()

    df['text_head'] = df[text_col].apply(lambda x:' '.join(x.split()[:50]))

    df['text_tail'] = df[text_col].apply(lambda x:' '.join(x.split()[-70:]))

    df['text_length'] = df[text_col].apply(lambda x:len(x.split()))

    df[text_col] = np.where(df['text_length']>120,

                            df['text_head'] + ' '+df['text_tail'],

                            df[text_col])

    df['text_length_2'] = df[text_col].apply(lambda x:len(x.split()))  

    

    return df



df_train = text_head_tail(df_train)

valid = text_head_tail(valid)

test = text_head_tail(test, 'content')



# 修改过长的原始text，只保留头尾的部分。（根据论文保留头尾是一个处理长文本分类问题的好解决方案）    
#x_train = regular_encode(train['comment_text'].values, tokenizer, maxlen=MAX_LEN)

x_train = regular_encode(df_train['comment_text'].values, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(valid['comment_text'].values, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test['content'].values, tokenizer, maxlen=MAX_LEN)



#y_train = train.toxic.values

y_train = df_train.toxic.values

y_valid = valid.toxic.values

# 调用tokenizer将text转化为tokens
# 将数据导入loader。

train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .shuffle(len(y_train))

    .batch(BATCH_SIZE)

    .repeat()

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
# 载入与训练的模型，注意这里使用了之前定义的MODEL变量（模型地址）。

with strategy.scope():

#     transformer_layer = TFAutoModel.from_pretrained('/kaggle/input/jigsaw-mlm-finetuned-xlm-r-large/')

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
# 训练training set。

n_steps = df_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    shuffle=False,

)
# 训练validation set。

# 此次比赛training set为纯英文，而validation set中含有其他语言，在validation set上训练可以提高模型对非英文的预测表现。

n_steps = valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=4,

    shuffle=False,

)
# 预测，储存模型，观察predictoin的分布。

sub['toxic'] = model.predict(x_test, verbose=1)

sub.to_csv('submission.csv', index=False)

model.save_weights('checkpoint.h5', overwrite=True)

sub.head(55)