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



seed = 42

np.random.seed(seed)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install googletrans textAugment
import tensorflow as tf

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



print("REPLICAS: ", strategy.num_replicas_in_sync)
df_train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

df_test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
from textaugment import EDA

from googletrans import Translator
for lang in df_train['language'].unique():

    number_of_training_samples = df_train[df_train['language'] == lang].shape[0] / df_train.shape[0]

    number_of_testing_samples = df_test[df_test['language'] == lang].shape[0] / df_test.shape[0]

    print('distribution of {} in training samples: {} and in testing samples {}'.format(lang, number_of_training_samples, number_of_testing_samples))
idx2label = {0: 'entailment', 1 :'neutral', 2: 'contradiction'}

for label in df_train['label'].unique():

    number_of_training_samples = df_train[df_train['label'] == label].shape[0] / df_train.shape[0]

    print('distribution of {} in training samples: {}'.format(idx2label[label], number_of_training_samples))

from transformers import BertTokenizer, TFBertModel

from transformers import RobertaTokenizer, TFRobertaModel

from transformers import TFXLMRobertaModel, XLMRobertaTokenizer

tf.random.set_seed(seed)
model_name = 'jplu/tf-xlm-roberta-large'

tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
import re



def clean_word(value):

    language = value[0]

    word = value[1]

    if language != 'English':

        word = word.lower()

        return word

    word = word.lower()

    word = re.sub(r'\?\?', 'e', word)

    word = re.sub('\.\.\.', '.', word)

    word = re.sub('\/', ' ', word)

    word = re.sub('--', ' ', word)

    word = re.sub('/\xad', '', word)

    word = word.strip(' ')

    return word



df_train['premise'] = df_train[['language', 'premise']].apply(lambda v: clean_word(v), axis=1)

df_train['hypothesis'] = df_train[['language', 'hypothesis']].apply(lambda v: clean_word(v), axis=1)

df_test['premise'] = df_test[['language', 'premise']].apply(lambda v: clean_word(v), axis=1)

df_test['hypothesis'] = df_test[['language', 'hypothesis']].apply(lambda v: clean_word(v), axis=1)
def build_model():

    with strategy.scope():

        bert_encoder = TFXLMRobertaModel.from_pretrained(model_name)

        input_word_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")

        input_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_mask")

        embedding = bert_encoder([input_word_ids, input_mask])[0]

        output_layer = tf.keras.layers.Dropout(0.25)(embedding)

        output_layer = tf.keras.layers.GlobalAveragePooling1D()(output_layer)

        output_dense_layer = tf.keras.layers.Dense(64, activation='relu')(output_layer)

        output_dense_layer = tf.keras.layers.Dense(32, activation='relu')(output_dense_layer)

        output = tf.keras.layers.Dense(3, activation='softmax')(output_dense_layer)



        model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=output)

        model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
with strategy.scope():

    model = build_model()

    model.summary()
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay



batch_size = 8 * strategy.num_replicas_in_sync

num_splits = 5

test_input = None
auto = tf.data.experimental.AUTOTUNE

languages = [ 'zh-cn' if lang == 'zh' else lang for lang in df_train['lang_abv'].unique()]



def make_dataset(train_input, train_label):

    dataset = tf.data.Dataset.from_tensor_slices(

        (

            train_input,

            train_label

        )

    ).repeat().shuffle(batch_size).batch(batch_size).prefetch(auto)

    return dataset
import multiprocessing as mp

from tqdm import tqdm_notebook



def xlm_roberta_encode(hypotheses, premises, src_langs, augmentation=False):

    num_examples = len(hypotheses)



    sentence_1 = [tokenizer.encode(s) for s in premises]

    sentence_2 = [tokenizer.encode(s) for s in hypotheses]

    input_word_ids = list(map(lambda x: x[0]+x[1], list(zip(sentence_1,sentence_2))))

    input_mask = [np.ones_like(x) for x in input_word_ids]

    inputs = {

        'input_word_ids': tf.keras.preprocessing.sequence.pad_sequences(input_word_ids, padding='post'),

        'input_mask': tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post')

    }

    return inputs
import gc

from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df_train, test_size=0.1)

if test_input is None:

    test_input = xlm_roberta_encode(df_test.hypothesis.values, df_test.premise.values, df_test.lang_abv.values,augmentation=False)

df_train['prediction'] = 0

num_augmentation = 1

train_input = xlm_roberta_encode(train_df.hypothesis.values,train_df.premise.values, train_df.lang_abv.values, augmentation=False)

train_label = train_df.label.values

train_sequence = make_dataset(train_input, train_label)

n_steps = (len(train_label)) // batch_size

validation_input = xlm_roberta_encode(validation_df.hypothesis.values, validation_df.premise.values,validation_df.lang_abv.values, augmentation=False)

validation_label = validation_df.label.values

tf.keras.backend.clear_session()

with strategy.scope():

    model = build_model()

    history = model.fit(

        train_sequence, shuffle=True, steps_per_epoch=n_steps, 

        validation_data = (validation_input, validation_label), epochs=50, verbose=1,

        callbacks=[

            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),

            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5),

            tf.keras.callbacks.ModelCheckpoint(

                'model.h5', monitor='val_accuracy', save_best_only=True,save_weights_only=True)

        ]

    )

model.load_weights('model.h5')

validation_predictions = model.predict(validation_input)

validation_predictions = np.argmax(validation_predictions, axis=-1)

validation_df['predictions'] = validation_predictions

acc = accuracy_score(validation_label, validation_predictions)

print('Accuracy: {}'.format(acc))

test_split_predictions = model.predict(test_input)

del train_input, train_label, validation_input, validation_label, model, train_sequence

gc.collect()
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt



fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15,10))

labels = list(idx2label.values())

for ax,language in zip(axes.flatten()[:validation_df['language'].nunique()],validation_df['language'].unique()):

    y_pred = validation_df[validation_df['language'] == language]['prediction'].values

    y_true = validation_df[validation_df['language'] == language]['label'].values

    lang_acc = accuracy_score(y_true, y_pred)

    print('Language {} has accuracy {}'.format(language, lang_acc))

    cm = confusion_matrix(np.array([idx2label[v] for v in y_true]), 

                      np.array([idx2label[v] for v in y_pred]), 

                      labels=labels)  

    cax = ConfusionMatrixDisplay(cm, display_labels=labels)

    cax.plot(ax=ax)

    ax.set_title(language)

    #fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)

    ax.set_yticklabels([''] + labels)

plt.title('Confusion matrix of the classifier')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.tight_layout()  

plt.show()
cm = confusion_matrix(np.array([idx2label[v] for v in validation_df.label.values]), 

                      np.array([idx2label[v] for v in validation_df.prediction.values]), 

                      labels=labels)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ConfusionMatrixDisplay(cm, display_labels=labels)

cax.plot(ax=ax)

plt.title('Confusion matrix of the classifier')

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
predictions = np.argmax(test_split_predictions, axis=-1)

submission = df_test.id.copy().to_frame()

submission['prediction'] = predictions
submission.head()

submission.to_csv("submission.csv", index = False)