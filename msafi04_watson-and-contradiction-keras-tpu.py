!pip install -q transformers==3.0.2
import numpy as np

import pandas as pd

import os

import gc

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm



os.environ["WANDB_API_KEY"] = "0"



import transformers

from transformers import AutoTokenizer, TFAutoModel



import tensorflow as tf

import tensorflow.keras.backend as K



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from time import time, strftime, gmtime



start = time()

#print(start)



import datetime

print(str(datetime.datetime.now()))



print(tf.version.VERSION)

print(transformers.__version__)
train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

train
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')

test
sub = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')

sub
plt.figure(figsize = (10, 10))

sns.countplot(train['label'])
lbls, freqs = np.unique(train['language'].values, return_counts = True)

#print(list(zip(lbls, freqs)))



plt.figure(figsize = (10, 10))

plt.title('Train')

plt.pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
lbls, freqs = np.unique(test['language'].values, return_counts = True)

#print(list(zip(lbls, freqs)))



plt.figure(figsize = (10, 10))

plt.title('Test')

plt.pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)



replicas = strategy.num_replicas_in_sync

batch_size = 8 * replicas

print("REPLICAS: ", strategy.num_replicas_in_sync)

print('Batch_size: ', batch_size)
model_name = 'jplu/tf-xlm-roberta-large'

epochs = 4

maxlen = 80



AUTO = tf.data.experimental.AUTOTUNE
tokenizer = AutoTokenizer.from_pretrained(model_name)
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize = (20, 15), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])

    #plt.show()
def get_training_dataset(idx, df = train, is_train = True):

    text = df[['premise', 'hypothesis']].values[idx].tolist()

    text_enc = tokenizer.batch_encode_plus(

                            text,

                            pad_to_max_length = True,

                            max_length = maxlen

                        )

    dataset = tf.data.Dataset.from_tensor_slices((text_enc['input_ids'], df['label'][idx].values))

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2020)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_valid_dataset(idx, df = train, is_train = False):

    text = df[['premise', 'hypothesis']].values[idx].tolist()

    text_enc = tokenizer.batch_encode_plus(

                            text,

                            pad_to_max_length = True,

                            max_length = maxlen

                        )

    dataset = tf.data.Dataset.from_tensor_slices((text_enc['input_ids'], df['label'][idx].values))

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset(df = test, is_train = False):

    text = df[['premise', 'hypothesis']].values.tolist()

    text_enc = tokenizer.batch_encode_plus(

                            text,

                            pad_to_max_length = True,

                            max_length = maxlen

                        )

    dataset = tf.data.Dataset.from_tensor_slices(text_enc['input_ids'])

    dataset = dataset.batch(batch_size)

    return dataset
def build_model(maxlen, model_name):

    with strategy.scope():

        #Load Transformer model

        base_model = TFAutoModel.from_pretrained(model_name)



        input_word_ids = tf.keras.Input(shape = (maxlen, ), dtype = tf.int32, name = "input_word_ids")



        #Encoding the input with the model

        embedding = base_model(input_word_ids)[0]



        #Extract the token used for classification, which is <s> and pass it to softmax (3 possible labels)

        out_tokens = embedding[:, 0, :]



        output = tf.keras.layers.Dense(3, activation = 'softmax')(out_tokens)



        model = tf.keras.Model(inputs = input_word_ids, outputs = output)



        model.compile(tf.keras.optimizers.Adam(lr = 1e-5), 

                      loss = 'sparse_categorical_crossentropy', 

                      metrics = ['accuracy'])

    

    return model
model = build_model(maxlen, model_name)

model.summary()
folds = 3

kf = KFold(n_splits = folds, shuffle = True, random_state = 777)

models = []

histories = []

predictions = np.zeros((test.shape[0], 3))



for fold, (trn_idx, val_idx) in enumerate(kf.split(np.arange(train['label'].shape[0]))):

    print('\n')

    print('-'*50)

    print(f'Training fold {fold + 1}')

    train_dataset = get_training_dataset(trn_idx, df = train, is_train = True)

    valid_dataset = get_valid_dataset(val_idx, df = train, is_train = False)

    K.clear_session()

    model = build_model(maxlen, model_name)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

                'XLM-R_fold-%i.h5'%fold, monitor = 'val_loss', verbose = 1, save_best_only = True,

                save_weights_only = True, mode = 'min', save_freq = 'epoch'

                )

    print('Model Training.....')

    STEPS_PER_EPOCH = len(trn_idx) // batch_size

    history = model.fit(

                train_dataset, epochs = epochs, verbose = 1, 

                steps_per_epoch = STEPS_PER_EPOCH,

                batch_size = batch_size, 

                validation_data = valid_dataset

            )

    

    display_training_curves(

                history.history['loss'], 

                history.history['val_loss'], 

                'loss', 311

                )

    display_training_curves(

                history.history['accuracy'], 

                history.history['val_accuracy'], 

                'accuracy', 312

                )

    histories.append(history)

    models.append(model)

    print('Prediting on test data..')

    test_dataset = get_test_dataset(test, is_train = False)

    pred = model.predict(test_dataset, verbose = 1)

    

    predictions += pred / folds

    

    del history, train_dataset, valid_dataset, model

    gc.collect()

print('\n')

print('-'*50)
sub['prediction'] = np.argmax(predictions, axis = 1)
sub.to_csv('./submission.csv', index = False)

sub
plt.figure(figsize = (10, 10))

sns.countplot(sub['prediction'])
finish = time()

print(strftime("%H:%M:%S", gmtime(finish - start)))