import pandas as pd

import numpy as np



from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import load_model

import keras

import pickle



import string

from joblib import Parallel, delayed

from tqdm import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

%matplotlib inline
import nltk

#nltk.download('stopwords')

#nltk.download('punkt')

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('english'))

stem = SnowballStemmer('english')
train_df = pd.read_csv("../input/train.csv")

train_df = train_df[['id','comment_text', 'target']]

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.target.hist()
train_df.shape
test_df.head()
test_df.shape
total_num_comments = train_df.shape[0]

unique_comments = train_df['comment_text'].nunique()



print('Train set: %d (Entries) and %d (Attributes).' % (train_df.shape[0], train_df.shape[1]))

print('Test set: %d (Entries) and %d (Attributes).' % (test_df.shape[0], test_df.shape[1]))



print('Number of Unique Comments {}'.format(unique_comments))

print('Percentage of Unique Comments %.2f%%' %( (unique_comments/total_num_comments)*100 ))
import matplotlib.pyplot as plt

import seaborn as sns

train_df['toxic_class'] = train_df['target'] >= 0.5

plt.figure(figsize=(16,4))

sns.countplot(train_df['toxic_class'])

plt.title('Toxic vs Non Toxic Comments')

plt.show()
train_data = train_df["comment_text"]

label_data = train_df["target"]

test_data = test_df["comment_text"]

train_data.shape, label_data.shape, test_data.shape
def tokenize(text):

    

    tokens = []

    for token in word_tokenize(text):

        if token in string.punctuation: continue

        if token in stop_words: continue

        tokens.append(stem.stem(token))

    

    return " ".join(tokens)
train_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in train_data.tolist())
train_tokens[0]
test_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in test_data.tolist())
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_tokens) + list(test_tokens))
train_data = tokenizer.texts_to_sequences(train_tokens)

test_data = tokenizer.texts_to_sequences(test_tokens)
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

from sklearn.model_selection import KFold

from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K

import numpy as np



X_train = train_data

y_train = label_data

X_test = test_data



KFold_N = 3

splits = list( KFold(n_splits=KFold_N).split(X_train,y_train) )



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
submission_df = pd.read_csv('../input/sample_submission.csv', index_col='id') 

submission_df['prediction'] = test_preds 

submission_df.reset_index(drop=False, inplace=True) 

submission_df.head()
submission_df.to_csv('submission.csv', index=False)