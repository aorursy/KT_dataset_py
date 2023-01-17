import numpy as np 

import pandas as pd 
train_embeddings = np.load('../input/cnn-embeddings-generator/embed_train_256_0.npy')

test_embeddings = np.load('../input/cnn-embeddings-generator/embed_test_256_0.npy')

external_train_embeddings = np.load('../input/cnn-embeddings-generator/embed_ext_2019_256_0.npy')

external_train_embeddings_18 = np.load('../input/cnn-embeddings-generator/embed_ext_2018_256_0.npy')

train_names = np.load('../input/cnn-embeddings-generator/names_train.npy')

test_names = np.load('../input/cnn-embeddings-generator/names_test.npy')

external_train_names = np.load('../input/cnn-embeddings-generator/names_ext_2019.npy')

train_labels = np.load('../input/cnn-embeddings-generator/labels_train.npy')

external_train_labels = np.load('../input/cnn-embeddings-generator/labels_ext_2019.npy')

external_train_labels_18 = np.load('../input/cnn-embeddings-generator/labels_ext_2018.npy')

train_embeddings.shape,test_embeddings.shape,external_train_embeddings.shape,train_names.shape,test_names.shape,external_train_names.shape
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
def build_model(dim=1280,lr=0.001):

    inp = tf.keras.layers.Input(shape=(None,dim))

    x = tf.keras.layers.Dense(1,activation='sigmoid')(inp)

    model = tf.keras.Model(inputs=inp,outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 

    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])

    return model



def build_model1(dim=1280,lr=0.001):

    inp = tf.keras.layers.Input(shape=(None,dim))

    x = tf.keras.layers.Dense(300,activation='sigmoid')(inp)

    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp,outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 

    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])

    return model
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report,accuracy_score,f1_score,roc_auc_score

skf = StratifiedKFold(n_splits=5,shuffle=True)



X = np.concatenate([train_embeddings,test_embeddings],axis=0)

y = np.zeros((train_embeddings.shape[0]+test_embeddings.shape[0],1))

y[:train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_test_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,test_embeddings],axis=0)

y = np.zeros((train_embeddings.shape[0]+test_embeddings.shape[0],1))

y[:train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_test1_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model1(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
import matplotlib.pyplot as plt

plt.hist(oof)

plt.show()
X = np.concatenate([external_train_embeddings,test_embeddings],axis=0)

y = np.zeros((external_train_embeddings.shape[0]+test_embeddings.shape[0],1))

y[:external_train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "ext_test_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
import matplotlib.pyplot as plt

plt.hist(oof)

plt.show()
X = np.concatenate([external_train_embeddings_18,test_embeddings],axis=0)

y = np.zeros((external_train_embeddings_18.shape[0]+test_embeddings.shape[0],1))

y[:external_train_embeddings_18.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "ext_test_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,external_train_embeddings],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings.shape[0],1))

y[:train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
import matplotlib.pyplot as plt

plt.hist(oof)

plt.show()
X = np.concatenate([train_embeddings,external_train_embeddings_18],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings_18.shape[0],1))

y[:train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([external_train_embeddings,external_train_embeddings_18],axis=0)

y = np.zeros((external_train_embeddings.shape[0]+external_train_embeddings_18.shape[0],1))

y[:external_train_embeddings.shape[0]] = 1

oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,external_train_embeddings],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings.shape[0],1))

y[:train_embeddings.shape[0]] = 1



labels = np.concatenate([train_labels,external_train_labels],axis=0)

X = X[labels==0]

y = y[labels==0]



oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,external_train_embeddings_18],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings_18.shape[0],1))

y[:train_embeddings.shape[0]] = 1



labels = np.concatenate([train_labels,external_train_labels_18],axis=0)

X = X[labels==0]

y = y[labels==0]



oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,external_train_embeddings],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings.shape[0],1))

y[:train_embeddings.shape[0]] = 1



labels = np.concatenate([train_labels,external_train_labels],axis=0)

X = X[labels==1]

y = y[labels==1]



oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))
X = np.concatenate([train_embeddings,external_train_embeddings_18],axis=0)

y = np.zeros((train_embeddings.shape[0]+external_train_embeddings_18.shape[0],1))

y[:train_embeddings.shape[0]] = 1



labels = np.concatenate([train_labels,external_train_labels_18],axis=0)

X = X[labels==1]

y = y[labels==1]



oof = np.zeros(y.shape)

print(X.shape,y.shape)



for i,(train_index, test_index) in enumerate(skf.split(X, y)):

    print("Fold:",i,end = " ")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_path = "train_ext_{}.h5".format(i)

    early_stop = EarlyStopping(monitor='val_auc',patience=20,verbose=1,mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4,verbose=0,mode='max')

    checkpoint = ModelCheckpoint(model_path , monitor='val_auc', verbose=0, save_best_only=True, mode='max')

    model = build_model(lr=0.01)

    history = model.fit(X_train,y_train,validation_data = (X_test,y_test),verbose=0,epochs=100, batch_size = 1000,

                        callbacks=[early_stop,reduce_lr,checkpoint])

    model.load_weights(model_path)

    oof[test_index] = model.predict(X_test)

    print("Partial Score:",roc_auc_score(y_test,oof[test_index]))

print(classification_report(y, (oof>0.5).astype(int), digits=4))

print(roc_auc_score(y, oof))