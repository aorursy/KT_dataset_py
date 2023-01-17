# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import keras.backend as K
from keras.utils import to_categorical
from keras.models import load_model

import os
print(os.listdir("../input"))
# Load train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
# reshape the datasets
train_X = train.drop('label', axis=1)
train_X = np.reshape(train_X.values, (train_X.shape[0], 28, 28, 1))
test_X = np.reshape(test.values, (test.shape[0], 28, 28, 1))

if K.image_data_format() == 'channels_first' :
    train_X = np.reshape(train_X.values, (train_X.shape[0], 1, 28, 28))
    test_X = np.reshape(test.values, (test.shape[0], 1, 28, 28))

train_Y = to_categorical(train['label'])

print('Train_X shape:', train_X.shape)
print('Train_Y shape:', train_Y.shape)
print('Test_X shape:', test_X.shape)
# Visualizing an image
plt.imshow(train_X[1,:,:,0])
print(train_Y[1].argmax())
model_path = 'model.hdf5'
try:
    print('Loading model -', model_path)
    model = load_model(model_path)
    print('Model successfully loaded!')
except:
    print("Can't find the model, creating a new one -", model_path)
    
    # create the model
    input_shape = (28,28,1) # number of channels, width, height
    if K.image_data_format() == 'channels_first' :
        input_shape = (1,28,28) # number of channels, width, height

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, (2,2), padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # train the model
    model.fit(train_X, train_Y, validation_split=0.2, batch_size=1000, epochs=20)
    model.save(model_path)
# evaluate model
#model.evaluate(train_X, train_Y) # prints loss and accuracy respectively
# make predictions
predictions = model.predict(test_X)
# test prediction
print(predictions[0].argmax())
plt.imshow(test_X[0,:,:,0])
# save final predictions by getting highest probabilities of the numbers
final_preds = []
for pred in predictions :
    final_preds.append(pred.argmax())
# Load sample submission file and insert our predictions into it
submission = pd.read_csv('../input/sample_submission.csv')
submission['Label'] = final_preds
# save submission file
submission.to_csv('final_submission.csv', index=False)

