import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import load_model

import keras

import pickle



import os

from tqdm import tqdm

tqdm.pandas()





import warnings

warnings.filterwarnings('ignore')
import os

print(os.listdir("../input"))
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
train.head()
test.head()
total_num_comments = train.shape[0]

unique_comments = train['comment_text'].nunique()





print('Train set: %d (Entries) and %d (Attributes).' % (train.shape[0], train.shape[1]))

print('Test set: %d (Entries) and %d (Attributes).' % (test.shape[0], test.shape[1]))



print('Number of Unique Comments {}'.format(unique_comments))

print('Percentage of Unique Comments %.2f%%' %( (unique_comments/total_num_comments)*100 ))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})
train['comment_length'] = train['comment_text'].apply(lambda x : len(x))

plt.figure(figsize=(16,4))

sns.distplot(train['comment_length'])

plt.show()
'''

test['comment_length'] = test['comment_text'].apply(lambda x : len(x))

plt.figure(figsize=(16,4))

sns.distplot(test['comment_length'])

plt.show()

'''
'''train['word_count'] = train['comment_text'].apply(lambda x : len(x.split(' ')))''

test['word_count'] = test['comment_text'].apply(lambda x : len(x.split(' ')))

bin_size = max(train['word_count'].max(), test['word_count'].max())//10

plt.figure(figsize=(20, 6))

sns.distplot(train['word_count'], bins=bin_size)

sns.distplot(test['word_count'], bins=bin_size)

plt.show()

'''
train['toxic_class'] = train['target'] >= 0.5

plt.figure(figsize=(16,4))

sns.countplot(train['toxic_class'])

plt.title('Toxic vs Non Toxic Comments')

plt.show()
'''train['created_date'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')

target_df = train.sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count'], 'target':['mean']})

target_df.columns = ['Date', 'Count', 'Toxicity Rate']'''
'''plt.figure(figsize=(16,4))

sns.lineplot(x=target_df['Date'], y=target_df['Toxicity Rate'])

plt.title('Toxicity over time')

plt.show()'''
'''plt.figure(figsize=(16,4))

sns.lineplot(x=target_df['Date'], y=target_df['Count'])

plt.title('Count of toxicity comments over time')

plt.show()'''
# Take care of dataframe memory

mem_usg = train.memory_usage().sum() / 1024**2 

print("Memory usage is: ", mem_usg, " MB")

train = train[["target", "comment_text"]]

mem_usg = train.memory_usage().sum() / 1024**2 

print("Memory usage is: ", mem_usg, " MB")
train_data = train["comment_text"]

label_data = train["target"]

test_data = test["comment_text"]

train_data.shape, label_data.shape, test_data.shape
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_data) + list(test_data))
train_data = tokenizer.texts_to_sequences(train_data)

test_data = tokenizer.texts_to_sequences(test_data)
MAX_LEN = 200

train_data = sequence.pad_sequences(train_data, maxlen=MAX_LEN)

test_data = sequence.pad_sequences(test_data, maxlen=MAX_LEN)
max_features = None
max_features = max_features or len(tokenizer.word_index) + 1

max_features
type(train_data), type(label_data.values), type(test_data)

label_data = label_data.values
# Keras Model

# Model Parameters

NUM_HIDDEN = 512

EMB_SIZE = 256

LABEL_SIZE = 1

MAX_FEATURES = max_features

DROP_OUT_RATE = 0.25

DENSE_ACTIVATION = "sigmoid"

NUM_EPOCHS = 1



# Optimization Parameters

BATCH_SIZE = 1024

LOSS_FUNC = "binary_crossentropy"

OPTIMIZER_FUNC = "adam"

METRICS = ["accuracy"]



class LSTMModel:

    

    def __init__(self):

        self.model = self.build_graph()

        self.compile_model()

    

    def build_graph(self):

        model = keras.models.Sequential([

            keras.layers.Embedding(MAX_FEATURES, EMB_SIZE),

            keras.layers.CuDNNLSTM(NUM_HIDDEN),

            keras.layers.Dropout(rate=DROP_OUT_RATE),

            keras.layers.Dense(LABEL_SIZE, activation=DENSE_ACTIVATION)])

        return model

    

    def compile_model(self):

        self.model.compile(

            loss=LOSS_FUNC,

            optimizer=OPTIMIZER_FUNC,

            metrics=METRICS)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import gc



X_train = train_data

y_train = label_data

X_test = test_data



KFold_N = 3

from sklearn.model_selection import KFold

splits = list( KFold(n_splits=KFold_N).split(X_train,y_train) )



from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K

import numpy as np





oof_preds = np.zeros((X_train.shape[0]))

test_preds = np.zeros((X_test.shape[0]))
for fold in range(KFold_N):

    K.clear_session()

    tr_ind, val_ind = splits[fold]

    ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    model = LSTMModel().model#build_model()

    model.fit(X_train[tr_ind],

        y_train[tr_ind]>0.5,

        batch_size=BATCH_SIZE,

        epochs=NUM_EPOCHS,

        validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

        callbacks = [es,ckpt])



    oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

    test_preds += model.predict(X_test)[:,0]

    

test_preds /= KFold_N    
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train>=0.5,oof_preds)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = test_preds

submission.reset_index(drop=False, inplace=True)

submission.head()
submission.to_csv('submission.csv', index=False)