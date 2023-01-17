import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Embedding, MaxPooling1D, Concatenate

from keras.layers import Input

from keras import backend as K 



import tensorflow as tf

from sklearn.metrics import roc_auc_score
base_data_dir = '../input/cat-in-the-dat/'

dtr = pd.read_csv(base_data_dir + "train.csv")

dts = pd.read_csv(base_data_dir + "test.csv")

dts.target = np.NaN

d = pd.concat([dtr, dts], sort=False)

train_set = dtr.shape[0]

del(dtr, dts)
d.columns
cat_features = [i for i in d.columns if not i in ("id","target")]

print(cat_features)
for c in cat_features:

    d[c] = d[c].astype("category")
cat_vectors = [d[c].cat.codes.to_numpy() for c in cat_features]
cat_size = [len(d[c].cat.categories) for c in cat_features]

cat_offset = np.cumsum([0] + cat_size[:-1])

cat_vectors2 = [cat_vectors[i] + cat_offset[i] for i in range(len(cat_vectors))]

cat_matrix = np.concatenate([np.reshape(np.ravel(c),(-1,1)) for c in cat_vectors2], axis=1)
print(cat_matrix.shape)

print(cat_matrix[0:2,])
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(range(train_set), test_size=0.2)
X_train = cat_matrix[train_idx,:]

X_test = cat_matrix[test_idx,:]

y_train = d.target.iloc[train_idx]

y_test = d.target.iloc[test_idx]

# X_train, X_test, y_train, y_test = train_test_split(cat_matrix[0:train_set,:], d.target[0:train_set], test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
max_features = np.max(cat_matrix)+1

maxlen = cat_matrix.shape[1]

print(max_features, maxlen)
# from https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

def auc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



embedding_size = 10



model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(10, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
print(model.summary())
hist = model.fit(X_train, y_train,validation_data=(X_test, y_test),

          batch_size=100, epochs=3, shuffle=True)
import matplotlib.pyplot as plt

plt.plot(hist.history['auc'])

plt.plot(hist.history['val_auc'])

plt.title('model ROC AUC')

plt.ylabel('AUC')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)

K.clear_session()
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)

plt.xlabel('False positive')

plt.ylabel('True positive')

plt.title('ROC auc='+str(auc(fpr, tpr)))

plt.show()
# from https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

def auc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



def prepare_one_cat_layer(c):

    inp = Input(shape=(1,))

    es = int(round(np.log(np.max(c))))+2 # just a guess

    emb = Embedding(np.max(c)+1, es, input_length=1)(inp)

    return (inp,emb)



cat_layers = [prepare_one_cat_layer(c) for c in cat_vectors]



x = Concatenate()([c[1] for c in cat_layers])

x = Flatten()(x)

x = Dropout(0.2)(x)

x = Dense(10, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(10, activation="relu")(x)

x = Dropout(0.2)(x)

final_layer = Dense(1, activation="sigmoid")(x)



model = Model(inputs=[c[0] for c in cat_layers], outputs=[final_layer])



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=[auc])
Xs_train = [c[train_idx] for c in cat_vectors]

Xs_test = [c[test_idx] for c in cat_vectors]
hist = model.fit(Xs_train, y_train, validation_data=(Xs_test, y_test),

          batch_size=100, epochs=3, shuffle=True)
import matplotlib.pyplot as plt

plt.plot(hist.history['auc'])

plt.plot(hist.history['val_auc'])

plt.title('model ROC AUC')

plt.ylabel('AUC')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(Xs_test)

K.clear_session()
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)

plt.xlabel('False positive')

plt.ylabel('True positive')

plt.title('ROC auc='+str(auc(fpr, tpr)))

plt.show()