import numpy as np

from keras.datasets import cifar10

from keras.utils import np_utils

import matplotlib.pyplot as plt

%matplotlib inline
# Load the dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape,X_test.shape)

print(y_train.shape, y_test.shape)
X_train = X_train[:1000]

y_train = y_train[:1000]

X_test = X_test[:10]

y_test = y_test[:10]



# There are 10 unique output classes 



# one hot encoding the labels

num_classes = 10

y_train = np_utils.to_categorical(y_train, num_classes)

y_test = np_utils.to_categorical(y_test, num_classes)
fig = plt.figure(figsize =(10,10))



for i in range(1,9):

    img = X_train[i-1]

    fig.add_subplot(2,4,i)

    plt.imshow(img)

    

print("Shape of each image:", X_train.shape[1:])
## Building classification model using keras



from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D

from keras.callbacks import callbacks



model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = X_train.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(32, (3,3), activation ='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64, (3,3), activation ='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(GlobalAveragePooling2D())

model.add(Dense(10, activation ='softmax'))



model.summary()
model.compile(loss='binary_crossentropy', optimizer ='adam', metrics =['accuracy'])
X_train_scaled  = X_train/255.

X_test_scaled = X_test/255.
## checkpointer will save best model



checkpointer = callbacks.ModelCheckpoint(filepath = "/kaggle/working/bestmodel.hdf5",verbose =1, save_best_only = True)
model.fit(X_train_scaled, y_train, batch_size = 32, epochs = 10, verbose =1, callbacks = [checkpointer], validation_split = 0.2, shuffle = True)
score = model.evaluate(X_test, y_test)



print("Accuracy on test dataset : ",score[1])
## Transfer learning - Resnet model 



from keras.applications.resnet50 import ResNet50,preprocess_input



model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (200,200,3))
# Reshaping trianing data as resnet model accepts minimum size of (197,197,3) so we are converting to (200,200,3)

#Reshaping the training data

#from scipy.misc import imresize

X_train_new = np.array([np.resize(X_train[i],(200,200,3)) for i in range(0,len(X_train))])



#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model. 

resnet_train_input = preprocess_input(X_train_new)



#Creating bottleneck features for the training data

train_features = model.predict(resnet_train_input)



#Saving the bottleneck features

np.savez('resnet_features_train', features=train_features)
## test data set resizing and preprocessing



from scipy.misc import imresize

X_test_new = np.array([np.resize(X_test[i],(200,200,3)) for i in range(0,len(X_test))])



#preprocess input for resnet

resnet_test_input = preprocess_input(X_test_new)



# creating bottleneck features for the training data

test_features = model.predict(resnet_test_input)



# saving the bottleneck features

np.savez('resnet_features_test', features = test_features)
model = Sequential()

model.add(GlobalAveragePooling2D(input_shape= train_features.shape[1:]))

model.add(Dropout(0.3))

model.add(Dense(10, activation ='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', 

              metrics=['accuracy'])
model.fit(train_features, y_train, batch_size=32, epochs=10,validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)
#Evaluate the model on the test data

score  = model.evaluate(test_features, y_test)



#Accuracy on test data

print('Accuracy on the Test Images: ', score[1])
## Due to memory issue while running resnet, reduced the number of input images from 60000 to 1000. Hence the accuracy is not that great.