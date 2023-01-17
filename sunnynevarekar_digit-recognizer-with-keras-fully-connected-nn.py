from __future__ import print_function



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop, Adam
batch_size = 128

num_classes = 10

epochs = 40
train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")

print(train_set.shape)

print(test_set.shape)
#separate labels and features

train_set_x = train_set.iloc[:, 1:785]

train_set_y = train_set.iloc[:, 0:1]



#convert pandas dataframe to numpy array

train_x = train_set_x.as_matrix()

train_y = train_set_y.as_matrix()



#no labels

test_x = test_set.as_matrix()



train_x = train_x.astype('float32')

test_x = test_x.astype('float32')



#feature scaling

train_x = train_x/255

test_x = test_x/255

digits, counts = np.unique(train_y, return_counts=True)

print(digits)

print(counts)

#convert class vectors to binary class metrices

train_y = keras.utils.to_categorical(train_y, num_classes)

#split train set into train and cross validation set

x_train, x_cv, y_train, y_cv = train_test_split(train_x, train_y, 

                                                train_size=40000, 

                                                random_state = 42,

                                                shuffle=True)
print(y_train.shape)

print(y_cv.shape)
#Define model for NN with 2 hidden layers

model = Sequential()



model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])
model.fit(x_train, y_train, 

          batch_size=batch_size, 

          epochs=epochs, 

          verbose=1, 

          validation_data=(x_cv, y_cv))
training_score = model.evaluate(x_train, y_train, verbose=0)

validation_score = model.evaluate(x_cv, y_cv, verbose=0)



print('Training loss', training_score[0])

print('Training accuracy', training_score[1])



print('Validation loss ', validation_score[0])

print('Validation accuracy ', validation_score[1])
predictions = model.predict(test_x)

predicted_labels = np.argmax(predictions, axis=1)
#Let's see 100 random images from the test set, each with its predicted label



permutations = np.random.permutation(28000)



fig, axs = plt.subplots(10, 10, figsize = (20, 20))

for r in range(10):

  for c in range(10):

    axs[r, c].imshow(np.reshape(test_x[permutations[10*r+c]]*255, (28, 28)), cmap='Greys')

    axs[r, c].axis('off')

    axs[r, c].set_title('prediction: '+str(predicted_labels[permutations[10*r+c]]))