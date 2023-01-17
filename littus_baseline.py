import pandas as pd

import numpy as np



from keras.utils import to_categorical

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

%matplotlib inline



FEATURES = [

    'total_acc_x_',

    'total_acc_y_',

    'total_acc_z_',



    'body_acc_x_',

    'body_acc_y_',

    'body_acc_z_',



    'body_gyro_x_',

    'body_gyro_y_',

    'body_gyro_z_'

]



TARGETS = [

    'WALKING',

    'WALKING_UPSTAIRS',

    'WALKING_DOWNSTAIRS',

    'SITTING',

    'STANDING',

    'LAYING'

]
train_x = np.fromfile('../input/con_train_x.bin').reshape((7352, 128, 9))

train_y = to_categorical(pd.read_csv('../input/con_train_y.csv')[['label']])

test_x = np.fromfile('../input/con_test_x.bin').reshape((2947, 128, 9))
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
pd.Series(dict(zip(TARGETS, train_y.sum(axis=0))))
X_train.shape # (sample, timestamp, feature_index)
test_x.shape
sample = X_train[0]



f, ax = plt.subplots(nrows=sample.shape[1], figsize=(16,15))



for i in range(sample.shape[1]):

    ax[i].set_title(FEATURES[i])

    ax[i].plot(sample[:, i])
from xgboost import XGBClassifier





xgb_train_x = X_train.reshape(len(X_train), -1)

xgb_val_x = X_val.reshape(len(X_val), -1)



xgb_train_y = np.argmax(y_train, axis=1)

xgb_val_y = np.argmax(y_val, axis=1)
xgbc = XGBClassifier(n_estimators=100, random_state=0, n_jobs=-1)
xgbc.fit(xgb_train_x, xgb_train_y)
preds = xgbc.predict(xgb_val_x)
accuracy_score(np.argmax(y_val, axis=1), preds)
xgb_train_x = train_x.reshape(len(train_x), -1)

xgb_test_x = test_x.reshape(len(test_x), -1)



xgb_train_y = np.argmax(train_y, axis=1)
xgbc.fit(xgb_train_x, xgb_train_y)
xgbc.predict(xgb_test_x)
from keras.models import Model

from keras.layers import Input, Dense, Add, Activation, Conv1D, GlobalAveragePooling1D

from keras.utils import np_utils

import numpy as np

import keras 

from keras.callbacks import ReduceLROnPlateau



 

def build_resnet(input_shape, n_feature_maps, nb_classes):    

    x = Input(shape=(input_shape))

    conv_x = keras.layers.normalization.BatchNormalization()(x)

    conv_x = keras.layers.Conv1D(n_feature_maps, 8, padding='same')(conv_x)

    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)

    conv_x = Activation('relu')(conv_x) 



    conv_y = keras.layers.Conv1D(n_feature_maps, 5, padding='same')(conv_x)

    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)

    conv_y = Activation('relu')(conv_y)



    conv_z = keras.layers.Conv1D(n_feature_maps, 3, padding='same')(conv_y)

    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)



    is_expand_channels = not (input_shape[-1] == n_feature_maps)

    if is_expand_channels:

        shortcut_y = keras.layers.Conv1D(n_feature_maps, 1, padding='same')(x)

        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    else:

        shortcut_y = keras.layers.normalization.BatchNormalization()(x)



    y = Add()([shortcut_y, conv_z])

    y = Activation('relu')(y)



    x1 = y

    conv_x = keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)

    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)

    conv_x = Activation('relu')(conv_x)





    conv_y = keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)

    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)

    conv_y = Activation('relu')(conv_y)





    conv_z = keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)

    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)



    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)

    if is_expand_channels:

        shortcut_y = keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)

        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    else:

        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)



    y = Add()([shortcut_y, conv_z])

    y = Activation('relu')(y)



    x1 = y

    conv_x = keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)

    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)

    conv_x = Activation('relu')(conv_x)



    conv_y = keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)

    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)

    conv_y = Activation('relu')(conv_y)



    conv_z = keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)

    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)



    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)

    if is_expand_channels:

        shortcut_y = keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)

        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    else:

        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)



    y = Add()([shortcut_y, conv_z])

    y = Activation('relu')(y)



    full = keras.layers.pooling.GlobalAveragePooling1D()(y)   

    out = Dense(nb_classes, activation='softmax')(full)



    return x, out
inputs, outputs = build_resnet(train_x.shape[1:], n_feature_maps=64, nb_classes=y_train.shape[1])
model = Model(inputs, outputs)

optimizer = keras.optimizers.Adam(lr=0.0001)



model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
chck = keras.callbacks.ModelCheckpoint("har_nn_bw.hdf", save_best_only=True)
history = model.fit(

    X_train, y_train, 

    validation_data=(X_val, y_val), epochs=10, batch_size=32, shuffle=True,

    callbacks=[chck]

)
pd.DataFrame({

    'loss': history.history['loss'],

    'val_loss': history.history['val_loss']

}).plot()
model.load_weights("har_nn_bw.hdf")
preds = model.predict(X_val)
accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds, axis=1))
test_y = pd.read_csv('../input/con_sample.csv')

test_y['label'] = np.argmax(model.predict(test_x), axis=1)



test_y.to_csv('submission.csv', index=False)