import numpy as np

import regex as re

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



os.environ['WANDB_API_KEY'] = '0' # to silence warning



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score



import tensorflow as tf

import tensorflow.keras.backend as K

import tokenizers

from transformers import *



import warnings

warnings.filterwarnings("ignore")
DATA_PATH = '/kaggle/input/contradictory-my-dear-watson-eng/'

MODEL_PATH = '/kaggle/input/tf-roberta/'
for dirname, _, filenames in os.walk(DATA_PATH):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.master())

except ValueError:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv(DATA_PATH + '/trans_train.csv', index_col=False)

test = pd.read_csv(DATA_PATH + '/trans_test.csv', index_col=False)
def basic_clean(row):

    row['hypothesis'] = re.sub(' +', ' ', row['hypothesis']).strip().lower()

    row['premise'] = re.sub(' +', ' ', row['premise']).strip().lower()

    return row



# Remove NaN rows

train = train[train['hypothesis'].notna()]

train = train[train['premise'].notna()]

test = test[test['hypothesis'].notna()]

test = test[test['premise'].notna()]



# Remove double spaces and starting/ending as well

train = train.apply(basic_clean, axis=1).reset_index()

test = test.apply(basic_clean, axis=1).reset_index()
MAX_LEN = 256



tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=MODEL_PATH + 'vocab-roberta-base.json', 

    merges_file=MODEL_PATH + 'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)
def encode_sentence(s):

    return tokenizer.encode(s)
encode_sentence('I love machine learning')
def roberta_encode(df, tokenizer):

    ct = df.shape[0]

    input_ids = np.ones((ct, MAX_LEN), dtype='int32')

    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')

    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')



    for k, row in df.iterrows():

        enc_hypothesis = tokenizer.encode(row['hypothesis'])

        enc_premise = tokenizer.encode(row['premise'])

        

        input_length = len(enc_hypothesis.ids) + len(enc_premise.ids) + 4

        if input_length > MAX_LEN:

            continue

        

        input_ids[k,:input_length] = [0] + enc_hypothesis.ids + [2,2] + enc_premise.ids + [2]

        

        attention_mask[k,:input_length] = 1

        

        type_sep = np.zeros_like([0])

        type_s1 = np.zeros_like(enc_hypothesis.ids)

        type_s2 = np.ones_like(enc_premise.ids)

        

        z = [type_sep, type_s1, type_sep, type_sep, type_s2, type_sep]

        token_type_ids[k,:input_length] = np.concatenate(z)



    return {

        'input_word_ids': input_ids,

        'input_mask': attention_mask,

        'input_type_ids': token_type_ids

    }
train_input = roberta_encode(train, tokenizer)

test_input = roberta_encode(test, tokenizer)
def build_model():

    input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')

    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')

    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')



    config = RobertaConfig.from_pretrained(MODEL_PATH + 'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(MODEL_PATH + 'pretrained-roberta-base.h5', config=config)

    x = bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

    

    # Huggingface transformers have multiple outputs, embeddings are the first one

    # let's slice out the first position

    x = x[0]

    

    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(1, 1)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(lr=1e-5),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy'])

    

    return model
with strategy.scope():

    model = build_model()

    model.summary()
accuracy = []

history = []



VER = 'v0'

EPOCHS = 4

KFOLDS = 5



# Our batch size will depend on number of replicas

BATCH_SIZE= 16 * strategy.num_replicas_in_sync



pred_test = np.zeros((test.shape[0], 3))



skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42)

for fold, (idxT, idxV) in enumerate(skf.split(train_input['input_word_ids'], train.label.values)):

    print('#' * 25)

    print('# FOLD %i' % (fold + 1))

    print('#' * 25)

    

    K.clear_session()

    with strategy.scope():

        print('Building model...')

        model = build_model()

        

        sv = tf.keras.callbacks.ModelCheckpoint(

            '%s-roberta-%i.h5' % (VER, fold),

            monitor='val_loss',

            verbose=1,

            save_best_only=True,

            save_weights_only=True,

            mode='auto',

            save_freq='epoch')



        kfold_train_input = {

            'input_word_ids': train_input['input_word_ids'][idxT,],

            'input_mask': train_input['input_mask'][idxT,],

            'input_type_ids': train_input['input_type_ids'][idxT,]}

        kfold_train_output = train.label.values[idxT,]

        

        kfold_val_input = {

            'input_word_ids': train_input['input_word_ids'][idxV,],

            'input_mask': train_input['input_mask'][idxV,],

            'input_type_ids': train_input['input_type_ids'][idxV,]}

        kfold_val_output = train.label.values[idxV,]



        print('Training...')

        kfold_history = model.fit(kfold_train_input,

                                  kfold_train_output,

                                  epochs=EPOCHS,

                                  batch_size=BATCH_SIZE,

                                  verbose=1,

                                  callbacks=[sv],

                                  validation_data=(kfold_val_input, kfold_val_output))

        history.append(kfold_history)



        print('Loading model...')

        model.load_weights('%s-roberta-%i.h5' % (VER, fold))



        # Compute prediction for this fold

        print('Predicting Test...')

        pred_test += model.predict(test_input) / skf.n_splits



        # Display fold accuracy

        print('Predicting OOF...')

        oof = [np.argmax(i) for i in model.predict(kfold_val_input)]

        kfold_accuracy = accuracy_score(oof, kfold_val_output)

        accuracy.append(kfold_accuracy)

        print('> FOLD %i - Accuracy: %.4f' % (fold + 1, kfold_accuracy))

        print()
print('> OVERALL KFold CV Accuracy: %.4f' % np.mean(accuracy))
plt.figure(figsize=(10, 10))

plt.title('Accuracy')



for i, hist in enumerate(history):

    xaxis = np.arange(len(hist.history['accuracy']))

    plt.subplot(3, 2, i + 1)

    plt.plot(xaxis, hist.history['accuracy'], label='Train set')

    plt.plot(xaxis, hist.history['val_accuracy'], label='Validation set')

    plt.gca().title.set_text('Fold %d accuracy curve' % (i + 1))

    plt.legend()
plt.figure(figsize=(10, 10))

plt.title('Loss')



for i, hist in enumerate(history):

    xaxis = np.arange(len(hist.history['accuracy']))

    plt.subplot(3, 2, i + 1)

    plt.plot(xaxis, hist.history['loss'], label='Train set')

    plt.plot(xaxis, hist.history['val_loss'], label='Validation set')

    plt.gca().title.set_text('Fold %d loss curve' % (i + 1))

    plt.legend()
submission = test.id.copy().to_frame()

submission['prediction'] = [np.argmax(i) for i in pred_test]
submission.head()
submission.to_csv('submission.csv', index=False)