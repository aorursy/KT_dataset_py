import pandas as pd
import numpy as np

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
tr_y = train['label']
tr_x = train.drop('label',axis=1)
tr_x = tr_x/255
test = test/255
tr_x = tr_x.values.reshape([-1,28,28,1])
test = test.values.reshape([-1,28,28,1])
tr_x[0].shape
from keras.utils import to_categorical
tr_y = to_categorical(tr_y)
from sklearn.model_selection import train_test_split
X = tr_x
y = tr_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
import tensorflow as tf
#conv2d hparams
input_shape = X_train.shape[1:]
strides = (3,3) #An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
filters = 32 # Number of kernels or feature detectors. 

#dense hparams
units = 1024 #dimension of output space. 

#dropout
rate = 0.33 #Dropout rate, 0-1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters, strides, padding='same', 
            activation='relu', input_shape=input_shape))    
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dropout(rate))

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(
    optimizer = tf.keras.optimizers.RMSprop(),
    loss = tf.keras.backend.categorical_crossentropy, 
    metrics = ['accuracy']
)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
history = model.fit(X_train, y_train, 
          batch_size=32,
          validation_data=(X_test, y_test),
          epochs=5, callbacks=[callback])
history.history
hist_df = pd.DataFrame(history.history)
hist_df[['loss','val_loss']].plot()
hist_df[['accuracy','val_accuracy']].plot()
results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)
submission.to_csv("submission.csv",index=False)