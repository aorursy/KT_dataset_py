import tensorflow as tf 

import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 

%matplotlib inline
#Import the input files

train = pd.read_csv('../input/digit-recognizer/train.csv') 

evaluation = pd.read_csv('../input/digit-recognizer/test.csv')

sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')



print(f'train shape = {train.shape}', f'test shape = {evaluation.shape}', sep='\n')
train.head()
print(train.isnull().any().sum())

print(evaluation.isnull().any().sum())
targets = train['label']

train = train.drop('label',axis = 1)
train.describe()
train /= 255

evaluation /= 255
train.describe()
index = np.random.randint(0,42000)

test_image = train.values[index].reshape(28,28)

plt.imshow(test_image, cmap = 'bone')

plt.title(targets.values[index])

plt.show()
train = train.values.reshape(-1,28,28,1)

evaluation = evaluation.values.reshape(-1,28,28,1)

targets = targets.values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, targets, stratify = targets, test_size = 0.1, random_state = 42)
from keras.models import Sequential

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten

from keras.optimizers import Adam
model = Sequential()



model.add(Conv2D(32, input_shape = (28,28,1), kernel_size = (3,3), activation = 'relu'))

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.1))



model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.1))





model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(128, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(10, activation = 'softmax'))



optimizer = Adam(lr=0.001)

model.compile(optimizer = optimizer,

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])



model.summary()
EPOCHS = 15

BATCH_SIZE = 256
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = EPOCHS, batch_size = BATCH_SIZE)
model.evaluate(X_test, y_test)
plt.figure(figsize=(9,6))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train','validation'])

plt.show()
evaluation = evaluation.reshape(28000,28,28,1)

results = model.predict_classes(evaluation)



results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)



submission.to_csv("submission.csv", index=False)