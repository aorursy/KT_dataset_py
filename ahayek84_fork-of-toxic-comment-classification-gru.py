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
!pip install tensorflow --upgrade
import numpy as np

np.random.seed(42)

import pandas as pd

import tensorflow as tf

from tensorflow import keras



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



import matplotlib.pyplot as plt 

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

os.environ['OMP_NUM_THREADS'] = '4'
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

local_sumb = pd.read_csv('../input/saved-relations/gru_we_submission.csv')

subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

test.head()
subm.head()
train.head()
text = train['comment_text']
text[0]
train['comment_text'][0]
# for train

lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
# for test

lens = test.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1) ## each colum may have the value of one ( Labled ) . 1- calc the max # if has no lable max = 0 then col = 1 -0 = 0

train.describe()
len(train),len(test)
## deal with nulls 

COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def clean(s): return re_tok.sub(r' \1 ', s)
## decide vocab size 

words = []

for t in text:

    words.extend(tokenize(t))

print(words[:100])

vocab = list(set(words))

print(len(words), len(vocab))
train['comment_text'][0]
clean(train['comment_text'][0])
def full_one_hot_word_embedding(vtrain_data,vtest_data):

    # switch data back to text 

    train_labels = vtrain_data[label_cols]

    txt_train_data = [clean(txt) for txt in train['comment_text']]

    txt_test_data = [clean(txt) for txt in test['comment_text']]

    

    # integer encode the documents

    vocab_size = 10000

    encoded_txt_train_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train_data]

    encoded_txt_test_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test_data]

    #print(encoded_txt_train_data)



    ptxt_train_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_train_data,

                                                            padding='post',

                                                            maxlen=5000)



    ptxt_test_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_test_data,

                                                           padding='post',

                                                           maxlen=5000)

    partial_x_train = ptxt_train_data[:100]

    partial_y_train = train_labels[:100]

    return (partial_x_train,partial_y_train,ptxt_test_data)
max_features = 30000

maxlen = 5000

embed_size = 300

vocab_size = 10000



def get_model():

    inp = keras.layers.Input(shape=(maxlen, ))

    x = keras.layers.Embedding(vocab_size, 16)(inp)

    x = keras.layers.SpatialDropout1D(0.2)(x)

    x = keras.layers.GRU(80, return_sequences=True)(x)

    # x = keras.layers.Bidirectional(keras.layers.LSTM(80, return_sequences=True))(x)

    avg_pool = keras.layers.GlobalAveragePooling1D()(x)

    max_pool = keras.layers.GlobalMaxPooling1D()(x)

    conc = keras.layers.concatenate([avg_pool, max_pool])

    outp = keras.layers.Dense(6, activation="sigmoid")(conc)

    

    model = keras.models.Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])



    return model
x_train,y_train,test_data = full_one_hot_word_embedding(train[:100],test[:100]) 
x_train.shape,y_train.shape
model1 = get_model()
batch_size = 32

epochs = 1



X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)



X_tra = X_tra.astype(np.float32)

X_val = X_val.astype(np.float32)

y_tra = y_tra.values.astype(np.float32)

y_val = y_val.values.astype(np.float32)
hist = model1.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=1)
# serialize model to JSON

model_json = model1.to_json()

with open("my_model1.json", "w") as json_file:

    json_file.write(model_json)

    

model1.save_weights('my_model1_weights.h5')
### load the model 1 

# load json and create model

json_file = open('my_model1.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model1 = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model

model1.load_weights("my_model1_weights.h5")

print("Loaded model from disk")
def column(matrix, i):

    return [row[i] for row in matrix]
##y_pred = model.predict(test_data, batch_size=1024)
submission = local_sumb
#submission = pd.DataFrame()

#submission['id'] = test['id']

#submission['toxic'] = column(y_pred, 0)

#submission['severe_toxic'] = column(y_pred, 1)

#submission['obscene'] = column(y_pred, 2)

#submission['threat'] = column(y_pred, 3)

#submission['insult'] = column(y_pred, 4)

#submission['identity_hate'] = column(y_pred, 5)
submission.to_csv('submission.csv', index=False)
submission