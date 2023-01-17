import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_data_origin=pd.read_csv("../input/train.csv")

test_data_origin=pd.read_csv("../input/test.csv")
train_data_origin.head()
print("Train: X =",train_data_origin.shape)

print("Test: X =",test_data_origin.shape)
train_features,train_labels = train_data_origin.drop("label",axis=1).values.astype('float32'),train_data_origin["label"].values.astype('int32') 

test_features = test_data_origin.values.astype('float32')
print("Train: X =",train_features.shape,"Y =",train_labels.shape)

print("Test: X =",test_features.shape)
import random

train_features_images=train_features.reshape(train_features.shape[0],28,28)

test_features_images=test_features.reshape(test_features.shape[0],28,28)



def show_images(features_images,labels,length):

    start=42

    for i in range(start, start+length):

        plt.subplot(330 + (i+1))

        plt.imshow(features_images[i], cmap=plt.get_cmap('gray'))

        plt.title(labels[i])

    plt.show()

        

show_images(train_features_images,train_labels,3)

show_images(test_features_images,np.zeros(test_features_images.shape[0]),3)

mean_px = train_features.mean().astype(np.float32)

std_px = train_features.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px



train_features_norm=standardize(train_features)

test_features_norm=standardize(test_features)
train_features_norm[:5]
train_features_reshaped=train_features_norm.reshape(train_features_norm.shape[0],28,28,1)

test_features_reshaped=test_features_norm.reshape(test_features_norm.shape[0],28,28,1)



train_labels_reshaped=train_labels.reshape(train_features.shape[0],1)

train_labels_reshaped=np.eye(10)[train_labels]
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(train_features_reshaped, train_labels_reshaped, test_size=0.10)

X_test=test_features_reshaped
print("Train: X =",X_train.shape,"Y =",y_train.shape)

print("Validation: X =",X_validation.shape,"Y =",y_validation.shape)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=X_train.shape[1:]))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 

                  metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint   



# train the model

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 

                               save_best_only=True)

hist = model.fit(X_train, y_train, batch_size=32, epochs=30,

          validation_data=(X_validation, y_validation), callbacks=[checkpointer], 

          verbose=2, shuffle=True)
# load the weights that yielded the best validation accurac

# NOTICE: by using modelcheckpoint you can early stop when you feel it overfitting

model.load_weights('model.weights.best.hdf5')
# evaluate and print test accuracy

score = model.evaluate(X_validation, y_validation, verbose=0)

print('\n', 'Test accuracy:', score[1])
X_test.shape
y_hat = model.predict(X_test)
y_hat[:5]
predict_labels=np.argmax(y_hat,axis=1)
import random

start=random.randint(0,len(X_test))

predict_labels[start:start+10]
evaluation= pd.DataFrame({'ImageId':np.arange(1,len(X_test)+1),'Label':predict_labels})

evaluation.tail()
evaluation.to_csv("evaluation_submission.csv",index=False)