import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from sklearn import metrics

from tqdm import tqdm

tqdm.pandas()



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



from keras.preprocessing.text import Tokenizer

from keras.preprocessing import text, sequence

from keras.models import load_model

import keras

from keras.models import Sequential

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Activation, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Flatten, GlobalMaxPooling1D

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



from sklearn.model_selection import train_test_split



import gc



# Any results you write to the current directory are saved as output.
TEXT_COL = 'comment_text'

EMB_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')
train_df.head()
train_df.shape
train_df.describe()
train_df.isna().sum()
train_df.columns
train_df.target.plot.hist()
train_df.target = np.where(train_df.target> 0.5, 1, 0)

print(train_df.target.value_counts())

sns.countplot(train_df.target)
#train_df['rating'].value_counts()

train_df['rating'] = np.where(train_df['rating'] == "approved", 1, 0)

train_df['rating'].value_counts()

sns.countplot(train_df['rating'])
features = ['severe_toxicity', 'obscene',

       'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual',

       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',

       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',

       'jewish', 'latino', 'male', 'muslim', 'other_disability',

       'other_gender', 'other_race_or_ethnicity', 'other_religion',

       'other_sexual_orientation', 'physical_disability',

       'psychiatric_or_mental_illness', 'transgender', 'white', 'rating', 'funny', 'wow',

       'sad', 'likes', 'disagree', 'sexual_explicit',

       'identity_annotator_count', 'toxicity_annotator_count']





toxicity_features = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]



identity_features = ["male", "female", "transgender", "other_gender", "heterosexual", "homosexual_gay_or_lesbian",

                     "bisexual", "other_sexual_orientation", "christian", "jewish", "muslim", "hindu", "buddhist",

                     "atheist", "other_religion", "black", "white", "asian", "latino", "other_race_or_ethnicity",

                     "physical_disability", "intellectual_or_learning_disability", "psychiatric_or_mental_illness", "other_disability"]



metadata_features = ["rating", "funny", "wow", "sad", "likes", "disagree", "toxicity_annotator_count", "identity_annotator_count"]
train_df[features].head()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(toxicity_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.title(col)

plt.tight_layout()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(identity_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.title(col)

plt.tight_layout()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(metadata_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.title(col)

plt.tight_layout()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(toxicity_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.hist(train_df[train_df["target"] == 0][col], alpha=0.5, label='0', color='b')

    plt.hist(train_df[train_df["target"] == 1][col], alpha=0.5, label='1', color='r') 

    plt.title(col)

plt.tight_layout()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(identity_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.hist(train_df[train_df["target"] == 0][col], alpha=0.5, label='0', color='b')

    plt.hist(train_df[train_df["target"] == 1][col], alpha=0.5, label='1', color='r') 

    plt.title(col)

plt.tight_layout()
print('Distributions columns')

plt.figure(figsize=(20, 150))

for i, col in enumerate(metadata_features):

    plt.subplot(40, 4, i + 1)

    plt.hist(train_df[col]) 

    plt.hist(train_df[train_df["target"] == 0][col], alpha=0.5, label='0', color='b')

    plt.hist(train_df[train_df["target"] == 1][col], alpha=0.5, label='1', color='r') 

    plt.title(col)

plt.tight_layout()
plt.close();

gc.collect();
train_data = train_df["comment_text"]

label_data = train_df["target"]

test_data = test_df["comment_text"]

train_data.shape, label_data.shape, test_data.shape
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_data) + list(test_data))
train_data = tokenizer.texts_to_sequences(train_df['comment_text'])

test_data = tokenizer.texts_to_sequences(test_df['comment_text'])
MAX_LEN = 200

train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)

test_data = sequence.pad_sequences(test_data, maxlen=MAX_LEN)



xtrain, xvalid, ytrain, yvalid = train_test_split(train_data, label_data, stratify=train_df.target, random_state=42, test_size=0.2, shuffle=True)
max_features = len(tokenizer.word_index) + 1

max_features
embedding_path1 = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

#embedding_path2 = "../input/glove840b300dtxt/glove.840B.300d.txt"

embed_size = 300



def get_coefs(word,*arr):

    return word, np.asarray(arr, dtype='float32')



def build_matrix(embedding_path, tokenizer):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



    word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.zeros((nb_words + 1, embed_size))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix



embedding_matrix = build_matrix(embedding_path1, tokenizer)
del train_data;

del train_df;

del test_df;

del tokenizer;

gc.collect();
def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
NUM_HIDDEN = 256

EMB_SIZE = 300

LABEL_SIZE = 1

MAX_FEATURES = max_features

DROP_OUT_RATE = 0.2

DENSE_ACTIVATION = "sigmoid"

NUM_EPOCH = 5

conv_size = 128



BATCH_SIZE = 512

LOSS_FUNC = "binary_crossentropy"

OPTIMIZER_FUNC = "adam"

METRICS = ["accuracy"]



from numpy.random import seed

seed(42)

from tensorflow import set_random_seed

set_random_seed(42)





model=Sequential()

model.add(Embedding(max_features, EMB_SIZE, weights=[embedding_matrix], trainable=False))

#model.add(keras.layers.Embedding(max_features, EMB_SIZE))

model.add(SpatialDropout1D(DROP_OUT_RATE))

model.add(LSTM(NUM_HIDDEN, return_sequences=True))

#model.add(Dropout(rate=DROP_OUT_RATE))

model.add(Conv1D(conv_size, 2, activation='relu', padding='same'))

model.add(MaxPooling1D(5, padding='same'))

model.add(Conv1D(conv_size, 3, activation='relu', padding='same'))

model.add(GlobalMaxPooling1D())

#model.add(Flatten())

model.add(Dense(LABEL_SIZE, activation=DENSE_ACTIVATION))



checkpointer = ModelCheckpoint(monitor='val_acc', mode='max', filepath='model.hdf5', verbose=2, save_best_only=True)

earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='max')



model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER_FUNC, metrics=METRICS)



history_lstm = model.fit(

    xtrain, 

    ytrain, 

    batch_size = BATCH_SIZE, 

    epochs = NUM_EPOCH, callbacks=[checkpointer, earlyStopping],

validation_data=(xvalid, yvalid))
plot_history(history_lstm)
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report



y_pred_lstm = model.predict_classes(xvalid, verbose=1, batch_size = BATCH_SIZE)

print(classification_report(yvalid, y_pred_lstm))



print()

print("accuracy_score", accuracy_score(yvalid, y_pred_lstm))



print()

print("Weighted Averaged validation metrics")

print("precision_score", precision_score(yvalid, y_pred_lstm, average='weighted'))

print("recall_score", recall_score(yvalid, y_pred_lstm, average='weighted'))

print("f1_score", f1_score(yvalid, y_pred_lstm, average='weighted'))



print()

from sklearn.metrics import confusion_matrix

import scikitplot as skplt

sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, y_pred_lstm)
submission_in = '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'

result = model.predict(test_data, verbose=1, batch_size = BATCH_SIZE)



submission = pd.read_csv(submission_in, index_col='id')

submission['prediction'] = result

submission.reset_index(drop=False, inplace=True)

submission.to_csv('submission.csv',index=False)