# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")

img_rows = 28

img_cols = 28

num_img = train.shape[0]

X_train = train.values[:,1:]

y_train = keras.utils.to_categorical(train.label, 10)

X_train = X_train.reshape(num_img, img_rows, img_cols, 1)/255.0



X_test = test.values[:]

test_num_img = test.shape[0]

X_test = X_test.reshape(test_num_img, img_rows, img_cols, 1)/255.0


model = Sequential()



model.add(Conv2D(256, kernel_size=(4, 4),

                 activation='elu',

                 input_shape=(img_rows, img_cols, 1)))



model.add(MaxPool2D(2, 2))



model.add(Conv2D(128, kernel_size=(3,3), activation='elu'))



model.add(MaxPool2D(2, 2))



model.add(Conv2D(128, kernel_size=(3,3), activation='elu'))



model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=(3,3), activation='elu'))





model.add(Flatten())



model.add(Dense(128,activation='elu'))

model.add(BatchNormalization())



model.add(Dropout(0.2))



model.add(Dense(128,activation='tanh'))

model.add(BatchNormalization())



model.add(Dense(10, activation='softmax'))
optimizer = keras.optimizers.Adam(learning_rate = 0.01)



reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor = 0.5, patience = 2, min_lr = 1e-6, verbose=1)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

checkpoint = ModelCheckpoint("",monitor='val_accuracy', verbose=1, save_best_only=True)



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit(X_train, y_train,

                    batch_size=128,

                    epochs=50,

                    validation_split = 0.1,

                    shuffle = True,

                    callbacks=[reduce_lr, earlystop, checkpoint])

model = keras.models.load_model("")

predictions = model.predict_classes(X_test)


out = pd.DataFrame({"ImageId": i+1 , "Label": predictions[i]} for i in range(0, test_num_img))

out.to_csv('submission.csv', index=False)