import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import BatchNormalization

import tensorflow.keras as keras

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from tensorflow.keras.preprocessing import sequence, text

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import model_from_json 

import os

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)





max_len = 1200

batch_size = 400*strategy.num_replicas_in_sync



epoch = 2

import matplotlib.pyplot as plt

# train = pd.read_csv('jigsaw-toxic-comment-train.csv')

# validation = pd.read_csv('validation.csv')

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

train.drop(['id', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

validation.drop(['id', 'lang'], axis=1, inplace=True)

X = train.append(validation)







X = X.sample(frac=1).reset_index(drop=True)

# size_data = 50000

# X = X.loc[:size_data, :]







# max length of string

print(train.comment_text.apply(lambda x: len(str(x).split())).max())

def roc_auc(predictions,target):

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc

xtrain, xvalid, ytrain, yvalid = train_test_split(X.comment_text.values, X.toxic.values, 

                                                stratify=X.toxic.values, 

                                                random_state=42, 

                                                test_size=0.2, shuffle=True)



# Tokenizer

token = text.Tokenizer(num_words=None)



token.fit_on_texts(list(xtrain) + list(xvalid))

xtrain_seq = token.texts_to_sequences(xtrain)

xvalid_seq = token.texts_to_sequences(xvalid)



#zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)



word_index = token.word_index



def create_model():

    model = Sequential()

    model.add(Embedding(len(word_index) + 1,

                    300,

                    input_length=max_len))

    model.add(SimpleRNN(100))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.TruePositives(name='tp'),

        keras.metrics.FalsePositives(name='fp'),

        keras.metrics.TrueNegatives(name='tn'),

        keras.metrics.FalseNegatives(name='fn'),

        keras.metrics.BinaryAccuracy(name='accuracy'),

        keras.metrics.Precision(name='precision'),

        keras.metrics.Recall(name='recall'),

        keras.metrics.AUC(name='auc')])

    return model

def save_model(model):

    model_json = model.to_json()

    with open("model.json", "w") as json_file:

        json_file.write(model_json)

    model.save_weights("model.h5")

    print("Saved model to disk")



def load_model():

    json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    print("Loaded model from disk")

    return loaded_model



weightList = list(train.toxic.value_counts())

toxic_weights = (1/weightList[1]) * ((weightList[1] + weightList[0])/2)

nontoxic_weights = (1/weightList[0]) * ((weightList[1] + weightList[0])/2)

class_weights = {0:nontoxic_weights, 1:toxic_weights}



if os.path.exists("model.h5") and os.path.exists('model.json'):

    model = load_model()

else:

    model = create_model()



print(model.summary())





model.fit(xtrain_pad, ytrain, epochs=epoch, batch_size=batch_size, class_weight=class_weights, validation_data=(xvalid_pad, yvalid))

save_model(model)
scores = model.predict(xvalid_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test.columns
test.head()

test_data = token.texts_to_sequences(test.content)

test_data_seq = sequence.pad_sequences(test_data, maxlen=max_len)
test['toxic'] = model.predict(test_data_seq, verbose=1)

test[['id', 'toxic']].to_csv('submission.csv', index=False)