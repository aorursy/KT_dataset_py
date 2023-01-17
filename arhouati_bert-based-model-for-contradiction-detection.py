# install need package
!pip install googletrans textAugment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # lib for gsraph plot
import os
import re # Regular Exprexion lib


os.environ["WANDB_API_KEY"] = "0" ## to silence warning

# lib for Machine learning models (BERT)
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from transformers import TFRobertaModel, RobertaTokenizer
import tensorflow as tf

from textaugment import EDA
from googletrans import Translator

import multiprocessing as mp
from tqdm import tqdm_notebook

import gc
from sklearn.model_selection import train_test_split
# TPU detection. No parameters necessary if TPU_NAME environment variable is
# set: this is always the case on Kaggle.
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU

print('Number of replicas:', strategy.num_replicas_in_sync)
# List of csv data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
df_test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
df_train.head()
labels, frequencies = np.unique(df_train.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.title('language distribution in Training Set')
plt.show()
labels, frequencies = np.unique(df_test.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.title('language distribution in Testing Set')
plt.show()
labels, freq_labels = np.unique(df_train.label.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(freq_labels,labels = labels, autopct = '%1.1f%%')
plt.title('labels distribution in Training Set')
plt.show()
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

languages = [ 'zh-cn' if lang == 'zh' else lang for lang in df_train['lang_abv'].unique()]
seed = 42
tf.random.set_seed(seed)

model_name = 'jplu/tf-xlm-roberta-large' # pretrained model' name
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name) # tokenizer init

#model_name = 'roberta-large'
#tokenizer = RobertaTokenizer.from_pretrained(model_name) # tokenizer init

def build_model():
    with strategy.scope():
        
        bert_encoder = TFXLMRobertaModel.from_pretrained(model_name)
        #bert_encoder = TFRobertaModel.from_pretrained(model_name)
        
        # define tensors for inputs
        input_word_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_mask")
        
        # Define model for fine-tuning XLMRoberta
        
        ### Layer 1 is a pretrained XLMRoberta Transformer
        embedding = bert_encoder([input_word_ids, input_mask])[0]
        
        ### 5 Layers before for Classification task
        output_layer = tf.keras.layers.Dropout(0.25)(embedding)
        output_layer = tf.keras.layers.GlobalAveragePooling1D()(output_layer)
        output_dense_layer = tf.keras.layers.Dense(64, activation='relu')(output_layer)
        output_dense_layer = tf.keras.layers.Dense(32, activation='relu')(output_dense_layer)
        output = tf.keras.layers.Dense(3, activation='softmax')(output_dense_layer)

        # Define Training parameters
        ## Optimizer is ADAM
        ## Function Loss is CrossEntropy
        ## Metric for evaluation is a standard accuracy
        model = tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=output)
        model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

# Init DeepLearning Model 
with strategy.scope():
    model = build_model()
    model.summary() # this describe model architecture and layers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

batch_size = 8 * strategy.num_replicas_in_sync
num_splits = 5
test_input = None
auto = tf.data.experimental.AUTOTUNE

def make_dataset(train_input, train_label):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            train_input,
            train_label
        )
    ).repeat().shuffle(batch_size).batch(batch_size).prefetch(auto)
    return dataset


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

# splite training data into train and valdiation
train_df, validation_df = train_test_split(df_train, test_size=0.1)

df_train['prediction'] = 0
num_augmentation = 1

# encoding training data
train_input = xlm_roberta_encode(train_df.hypothesis.values,train_df.premise.values, train_df.lang_abv.values, augmentation=False)
train_label = train_df.label.values

# create data Iterator for training 
train_sequence = make_dataset(train_input, train_label)

# encoding validation data
validation_input = xlm_roberta_encode(validation_df.hypothesis.values, validation_df.premise.values,validation_df.lang_abv.values, augmentation=False)
validation_label = validation_df.label.values
tf.keras.backend.clear_session()
# splite training data into train and valdiation
train_df, validation_df = train_test_split(df_train, test_size=0.1)

df_train['prediction'] = 0
num_augmentation = 1

# encoding training data
train_input = xlm_roberta_encode(train_df.hypothesis.values,train_df.premise.values, train_df.lang_abv.values, augmentation=False)
train_label = train_df.label.values

# create data Iterator for training 
train_sequence = make_dataset(train_input, train_label)

# encoding validation data
validation_input = xlm_roberta_encode(validation_df.hypothesis.values, validation_df.premise.values,validation_df.lang_abv.values, augmentation=False)
validation_label = validation_df.label.values
tf.keras.backend.clear_session()
n_steps = (len(train_label)) // batch_size

with strategy.scope():
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

# save trained model
model.load_weights('model.h5')

# calcul of validation Accuracy
validation_predictions = model.predict(validation_input)
validation_predictions = np.argmax(validation_predictions, axis=-1)
validation_df['predictions'] = validation_predictions
acc = accuracy_score(validation_label, validation_predictions)
print('Accuracy: {}'.format(acc))
# encoding test data for prediction and submission
if test_input is None:
    test_input = xlm_roberta_encode(df_test.hypothesis.values, df_test.premise.values, df_test.lang_abv.values,augmentation=False)

# prediction using trained model
test_split_predictions = model.predict(test_input)
predictions = np.argmax(test_split_predictions, axis=-1)

# create submission file
submission = df_test.id.copy().to_frame()
submission['prediction'] = predictions
submission.head()

# submission to challenge
submission.to_csv("submission.csv", index = False)