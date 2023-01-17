import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import classification_report

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, SGD, Adam,TFOptimizer

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
target = train["label"]

features = train.drop(labels = ["label"],axis = 1) 
# Scaling the data

features = features / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28, width = 28 , channel = 1)

features = features.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors

target = to_categorical(target, num_classes = 10)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.1)
# Create a function to build our classifier

clf = Sequential()

clf.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                     activation ='relu', input_shape = (28,28,1)))

clf.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                     activation ='relu'))

clf.add(MaxPool2D(pool_size=(2,2)))

clf.add(Dropout(0.25))



clf.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

clf.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

clf.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

clf.add(Dropout(0.3))



clf.add(Flatten())

clf.add(Dense(512, activation = "relu", use_bias= True))

clf.add(Dropout(0.5))

clf.add(Dense(10, activation = "softmax"))



clf.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Data augmentation to prevent overfitting



datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

clf.fit_generator(datagen.flow(X_train,Y_train, batch_size= 82),

                              epochs = 30, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // 82

                              , callbacks=[learning_rate_reduction])
# predict results

results = clf.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnnmodel.csv",index=False)