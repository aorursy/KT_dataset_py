import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt
train_df=pd.read_csv('../input/train.csv')

print(train_df.shape)

train_df.sample(5)
test_df=pd.read_csv('../input/test.csv')

print(test_df.shape)

test_df.sample(5)
# the first column = labels (0,1,2, .....,9)

Y_train=train_df.iloc[:, 0].values

#Y_train=train_df['label']



# second to last column = 28x28 = 784 pixel data

X_train=train_df.iloc[:, 1:].values



print("training data: {}".format(X_train.shape))

print("training labels {}".format(Y_train.shape))
X_test=test_df.values

print("test data: {}".format(X_test.shape))
img_width=28

img_height=28

img_depth=1



#reshape the data

X_train = X_train.reshape(len(X_train),img_width,img_height,img_depth)

X_test = X_test.reshape(len(X_test),img_width,img_height,img_depth)



#convert to floating point number

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



#Normalize pixel values

X_train /= 255

X_test /= 255



print("Training matrix shape", X_train.shape)

print("Testing matrix shape", X_test.shape)
plt.figure(figsize=(12,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Class {}".format(Y_train[i]))
from keras.utils.np_utils import to_categorical

Y_train= to_categorical(Y_train, num_classes=10)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D





model = Sequential()

model.add(Convolution2D(32,(3, 3), activation='relu', input_shape=(28,28, 1)))

model.add(Convolution2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.25))

 

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

saved_path='cnn_model.h5'

model_checkpoint = ModelCheckpoint(filepath=saved_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=6, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001, mode='auto')

callback_list = [model_checkpoint, early_stop, reduce_lr]



train_history = model.fit(X_train, Y_train, batch_size=32, 

                          epochs=15, verbose=1, callbacks=callback_list, validation_split=0.2)
from keras.models import load_model

model=load_model('cnn_model.h5')

labels=model.predict_classes(X_test)
plt.figure(figsize=(12,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[i].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("predicted class {}".format(labels[i]))
submission=open("submission.csv", 'w')

submission.write("ImageId,Label\n")

for i in range(len(labels)):

    submission.write("{},{}\n". format(i+1, labels[i]))

    

from IPython.display import FileLink

FileLink("submission.csv")