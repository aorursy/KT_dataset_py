import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

from keras.utils import np_utils



# Read data

train = pd.read_csv('../input/train.csv')

y = train['label'].values

y = np_utils.to_categorical(y)

X = train[train.columns[1:]].values

X_test = pd.read_csv('../input/test.csv').values



# split data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)

print("Shape of X_train: {}".format(X_train.shape))

print("Shape of y_train: {}".format(y_train.shape))

print("Shape of X_val: {}".format(X_val.shape))

print("Shape of y_val: {}".format(y_val.shape))
X_train = X_train / 255

X_val = X_val / 255

X_test = X_test / 255
from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Activation

from keras.layers.core import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import callbacks

from keras.optimizers import Adagrad





model = Sequential()



model.add(Conv2D(100, kernel_size=(5,5), activation='elu', padding='same', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, kernel_size=(5,5), activation='elu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

for j in range(12):

    model.add(Dense(100, use_bias=False))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    model.add(Activation('elu'))   



model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=Adagrad(), metrics=['accuracy'])



model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=50, batch_size=32, 

          validation_data=(X_val.reshape(-1, 28, 28, 1), y_val),

          callbacks=[callbacks.TerminateOnNaN(), callbacks.EarlyStopping(patience=3)])
scores = model.evaluate(X_val.reshape(-1, 28, 28, 1), y_val)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_pred = model.predict(X_val.reshape(-1, 28, 28, 1))

print(y_pred.shape)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import balanced_accuracy_score



y_pred_class = (y_pred >= 0.5).astype(float).argmax(axis=1);

y_val_class = y_val.argmax(axis=1)



print(confusion_matrix(y_val_class, y_pred_class))

print(balanced_accuracy_score(y_val_class, y_pred_class))
y_pred_test = model.predict(X_test.reshape(-1, 28, 28, 1))

y_test_class = y_pred_test.argmax(axis=1)
# output result

dataframe = pd.DataFrame({"ImageId": list(range(1,len(y_test_class)+1)), "Label": y_test_class})

dataframe.to_csv('output_nn.csv', index=False, header=True)