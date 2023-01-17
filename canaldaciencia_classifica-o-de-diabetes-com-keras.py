from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint



seed = 42

np.random.seed(seed)


df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.head()
# First - split into Train/Test

from sklearn.model_selection import train_test_split



X = df.drop(['Outcome'],axis=1)



y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



print(X_train.shape)

print(X_test.shape)
# Ensure that fieldnames aren't included

X_train = X_train.values

y_train = y_train.values

X_test  = X_test.values

y_test  = y_test.values
NB_EPOCHS = 1000  # num of epochs to test for

BATCH_SIZE = 16



## Create our model

model = Sequential()



# 1st layer: input_dim=8, 12 nodes, RELU

model.add(Dense(12, input_dim=8, activation='relu'))

# 2nd layer: 8 nodes, RELU

model.add(Dense(8, activation='relu'))

# output layer: dim=1, activation sigmoid

model.add(Dense(1, activation='sigmoid' ))



# Compile the model

model.compile(loss='binary_crossentropy',   # since we are predicting 0/1

             optimizer='adam',

             metrics=['accuracy'])



# checkpoint: store the best model

ckpt_model = 'pima-weights.best.hdf5'

checkpoint = ModelCheckpoint(ckpt_model, 

                            monitor='val_acc',

                            verbose=1,

                            save_best_only=True,

                            mode='max')

callbacks_list = [checkpoint]



print('Starting training...')

# train the model, store the results for plotting

history = model.fit(X_train,

                    y_train,

                    validation_data=(X_test, y_test),

                    epochs=NB_EPOCHS,

                    batch_size=BATCH_SIZE,

                    callbacks=callbacks_list,

                    verbose=2)
# Model accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Model Losss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# print final accuracy

scores = model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))