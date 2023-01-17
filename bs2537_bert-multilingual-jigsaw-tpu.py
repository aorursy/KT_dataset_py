

import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import tensorflow as tf

print("Tensorflow version " + tf.__version__)



try:

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection

  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

except ValueError:

  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')



tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re
train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv')
train.shape
train.isnull().sum()
#Data cleaning



import re



def text_process(text):

    '''Make text lowercase, remove text in square brackets,remove all single character, Substituting multiple spaces with single space,removing all special characters,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]#', '', text)

    #text = re.sub(r'\W', ' ', str(X[text]))

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub(r'\s+', ' ', text, flags=re.I)

    #text = re.sub('[%s]' % re.escape(str.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('©', '', text)

    text = re.sub('@', '', text)

    text = re.sub('#', '', text)

    text = re.sub('ûò', '', text)

    text = re.sub('!', '', text)

    text = re.sub('&', '', text)

 

    #text = re.sub('?', '', text)

    return text
train['comment_text'] = train['comment_text'].apply(lambda x: text_process(x))
train = train.rename(columns={"comment_text": "text"})
#Validation dataset



valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv')
valid.head()
# Applying the cleaning function 

valid['comment_text'] = valid['comment_text'].apply(lambda x: text_process(x))
valid = valid.rename(columns={"comment_text": "text"})
#Load test dataset



test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv')
test.head()
# Applying the cleaning function 

test['comment_text'] = test['comment_text'].apply(lambda x: text_process(x))
test = test.rename(columns={"comment_text": "text"})
!pip install transformers
#Bert Model



import tensorflow

from tensorflow import keras

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam, Nadam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
#Function for encoding the word/comment to integer or vector



def regular_encode(texts, tokenizer, maxlen=512):

  enc_di = tokenizer.batch_encode_plus(

      texts,

      return_attention_masks=False,

      return_token_type_ids=False,

      pad_to_max_length=True,

      max_length=maxlen

  )



  return np.array(enc_di['input_ids'])
#BUILD MODEL FUNCTION



def build_model(transformer, max_len=512):

  input_word_ids = Input(shape=max_len, dtype=tf.int32, name='input_word_ids')

  sequence_output=transformer(input_word_ids)[0]

  cls_token=sequence_output[:, 0, :]

  out = Dense(1, activation='sigmoid')(cls_token)



  model = Model(inputs=input_word_ids, outputs=out)

  model.compile(Adam(lr=1e-5), loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC()])



  return model
EPOCHS = 10

BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync 

MAX_LEN = 512  

MODEL = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
#Encode comments (text into vectors)



x_train = regular_encode(train.text.values, tokenizer, maxlen=MAX_LEN)



x_valid = regular_encode(valid.text.values, tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values



y_valid = valid.toxic.values
x_test = regular_encode(test.text.values, tokenizer, maxlen=MAX_LEN)
AUTO = tf.data.experimental.AUTOTUNE
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)    

    .batch(BATCH_SIZE)

    .prefetch(AUTO) 

)
valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))  #no shuffle or repeat

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test) 

    .batch(BATCH_SIZE)

)
#Model
MODEL = 'bert-base-multilingual-cased'
#Build model from BERT pretrained model and the build model function



with tpu_strategy.scope():   #enables to use the TPU while training the model

  transformer_layer = TFAutoModel.from_pretrained(MODEL)

  model = build_model(transformer_layer, max_len = MAX_LEN)
model.summary()
!pip install h5py
#Call backs



EPOCHS = 2



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler



#stop = EarlyStopping(monitor='val_auc', mode='max', min_delta=0.01, patience=1, verbose=1)



#rlrop = ReduceLROnPlateau(monitor='val_auc', mode='auto', min_delta=0.01, factor=0.2, patience=1) 



#filepath = '/kaggle/working/bert_model_tpu_v2.hdf5'   # Saved model checkpoint file path



#checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max', save_freq='epoch')
start_lr = 0.00001

min_lr = 0.00001

max_lr = 0.00005 * tpu_strategy.num_replicas_in_sync



rampup_epochs = 5

sustain_epochs = 0

exp_decay=0.8



def lrfn(epoch):

  if epoch < rampup_epochs:

    return (max_lr - start_lr)/rampup_epochs * epoch + start_lr

  elif epoch < rampup_epochs + sustain_epochs:

    return max_lr

  else:

    return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr



lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)



rang = np.arange(EPOCHS)

y = [lrfn(x) for x in rang]

plt.plot(rang, y)



print('Learning rate per epoch:')
#Model training



n_steps = x_train.shape[0] // BATCH_SIZE



model.fit(train_dataset, steps_per_epoch=n_steps, validation_data=valid_dataset, epochs=2, callbacks=[lr_callback])
n_steps = x_valid.shape[0] // BATCH_SIZE

model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=2

)
sub = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)