# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB,GaussianNB



from keras.models import Sequential

from keras.layers import Dense, Dropout,Conv2D, Flatten

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils



from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold



## Keras imports

from keras.callbacks import Callback,ModelCheckpoint

from keras.models import Sequential,load_model

from keras.layers import Dense, Dropout,LSTM

from keras.wrappers.scikit_learn import KerasClassifier

import keras.backend as K

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding,Conv1D,Dropout,GlobalMaxPool1D,Dense,Activation

import tensorflow as tf



# Any results you write to the current directory are saved as output.



## Visualization Library

from IPython.display import display

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub.head()
print(train.head())

print("==")

print(train.shape)

print("===")

print(test.head())

print("==")

print(test.shape)
train['target'].value_counts(normalize=True)
independent_feat = ['var_'+str(i) for i in range(200)]

dependent_feat = 'target'
train[independent_feat].describe()
sc = StandardScaler()

train[independent_feat] = sc.fit_transform(train[independent_feat])

test[independent_feat] = sc.transform(test[independent_feat])
train, val = train_test_split(train, test_size=0.20, stratify=train[dependent_feat])

print(train.shape)

print(val.shape)
def plot_history(history):

    auc = history.history['get_auc']

    val_auc = history.history['val_get_auc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(auc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, auc, 'b', label='Training AUC')

    plt.plot(x, val_auc, 'r', label='Validation AUC')

    plt.title('Training and validation AUC')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



def get_auc(y_true, y_pred):

    print("True value {}".format(y_true))

    print("Predicted value {}".format(y_pred))

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
def get_cnn_data(data):

    return data.values.reshape(data.shape[0], 20, 10, 1)



train_cnn = get_cnn_data(train[independent_feat])

val_cnn = get_cnn_data(val[independent_feat])

test_cnn = get_cnn_data(test[independent_feat])
## Define Model

input_shape = len(independent_feat)



def cnn_v1():

    model = Sequential()

    #add model layers

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(20,10,1)))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(Conv2D(16, kernel_size=3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    #compile model using accuracy to measure model performance

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[get_auc])

    print(model.summary())

    return model

mode_path = 'cnn_v1.h5'

callbacks = [ModelCheckpoint(filepath=mode_path, save_best_only=True)]
## Run Model

print("Compile model ...")

estimator = KerasClassifier(build_fn=cnn_v1, epochs=10, batch_size=64)

history = estimator.fit(train_cnn, train[dependent_feat],\

                        validation_data=(val_cnn,val[dependent_feat]),callbacks=callbacks)
plot_history(history)
# https://github.com/keras-team/keras/issues/5916

model = load_model(mode_path,custom_objects={'get_auc':get_auc})
train_pred = model.predict_proba(train_cnn)

val_pred = model.predict_proba(val_cnn)
train_pred = [i[0] for i in train_pred]

val_pred = [i[0] for i in val_pred]
from sklearn.metrics import roc_auc_score

print("Train auc {}".format(roc_auc_score(train[dependent_feat],train_pred)))

print("Val auc {}".format(roc_auc_score(val[dependent_feat],val_pred)))
test_pred = model.predict_proba(test_cnn)

test_pred = [i[0] for i in test_pred]
result = pd.DataFrame({'ID_code':test['ID_code'],'target':list(test_pred)})

result.head()
result.to_csv('cnn_v1.csv',index=False)