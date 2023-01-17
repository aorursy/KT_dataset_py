import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
np.random.seed(42)
EPOCH = 200
BATCH_SIZE = 1
VERBOSE = 1
NB_CLASSES = 2
N_HIDDEN = 64
VALIDATION_SPLIT = 0.2
data = pd.read_csv('../input/gender-classification/Transformed Data Set - Sheet1.csv')
data.columns
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data.iloc[:,:-1]).toarray()
RESHAPED = feature_arr.shape[1]
RESHAPED
feature_arr
X = feature_arr
y = np.array(data.iloc[:,-1])
y = LabelEncoder().fit_transform(y)
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_T = X_train.reshape(X_train.shape[0], RESHAPED)
X_test_T = X_test.reshape(X_test.shape[0], RESHAPED)
X_test_T.shape
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)
model2 = tf.keras.models.Sequential()
model2.add(keras.layers.Dense(1024, input_shape = (RESHAPED,), name='dense_layer', activation='relu'))
model2.add(keras.layers.Dense(512, name = 'dense_layer_2', activation='relu'))
model2.add(keras.layers.Dense(256, name = 'dense_layer_3', activation='relu'))
model2.add(keras.layers.Dense(NB_CLASSES, name = 'dense_layer_4', activation='softmax'))
model2.summary()
model2.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(X_train, y_train ,batch_size =BATCH_SIZE, epochs =  10, verbose = VERBOSE, validation_split = VALIDATION_SPLIT)
test_loss, test_acc = model2.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
DROPOUT = 0.5
model3 = tf.keras.models.Sequential()
model3.add(keras.layers.Dense(1024,
          input_shape=(RESHAPED,),
          name='dense_layer', activation='relu', ))
model3.add(keras.layers.Dropout(DROPOUT))
model3.add(keras.layers.Dense(512,
          name='dense_layer_2', activation='relu', ))
model3.add(keras.layers.Dropout(DROPOUT))
model3.add(keras.layers.Dense(256,
          name='dense_layer_3', activation='relu', ))
model3.add(keras.layers.Dropout(DROPOUT))
model3.add(keras.layers.Dense(NB_CLASSES,
          name='dense_layer_4', activation='softmax'))
model3.summary()
model3.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model3.fit(X_train, y_train,
          batch_size=BATCH_SIZE, epochs=10,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
test_loss_3, test_acc_3 = model2.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)