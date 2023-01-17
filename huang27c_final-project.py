import pandas as pd

import numpy as np

import matplotlib.image as mpimg

import time

from sklearn.model_selection import train_test_split



cactus_label = pd.read_csv('../input/train.csv')



#read in training set

train_img = []

train_lb = []

for i in range(len(cactus_label)):

    row = cactus_label.iloc[i]

    fileName = row['id']

    train_lb.append(row['has_cactus'])

    path = "../input/train/train/{}".format(fileName)

    im = mpimg.imread(path)

    train_img.append(im)

    

X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 

X_train = np.array(X_train)

X_test = np.array(X_test)
import os

test_img = []

sample = pd.read_csv('../input/sample_submission.csv')

folder = '../input/test/test/'

                   

for i in range(len(sample)):

    row = sample.iloc[i]

    fileName = row['id']

    path = folder + fileName

    img = mpimg.imread(path)

    test_img.append(img)

                     

test_img = np.asarray(test_img)
cactus_label['has_cactus'].value_counts()
import matplotlib.pyplot as plt

# Data to plot

labels = 'Has Cactus', 'No Cacuts'

sizes = [13136, 4364]

colors = ['yellowgreen', 'lightskyblue']

 

# Plot

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
'''

Convert 3D ararys to 1D array

Paramter: a list of 3D images

Return: a list of 1D images

'''

def imageToFeatureVector(images):

    flatten_img = []

    for img in images:

        data = np.array(img)

        flattened = data.flatten()

        flatten_img.append(flattened)

    return flatten_img
from sklearn.neighbors import KNeighborsClassifier

start = time.time()



X_train_flatten = imageToFeatureVector(X_train)

X_test_flatten = imageToFeatureVector(X_test)



knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_flatten, y_train) 

score = knn.score(X_test_flatten, y_test)



end = time.time()

print("The run time of KNN is {:.3f} seconds".format(end-start))

print("KNN alogirthm's test score is: {:.3f}".format(score))
import pandas as pd

import numpy as np

import matplotlib.image as mpimg

import time

from sklearn.model_selection import train_test_split

cactus_label = pd.read_csv('../input/train.csv')



#read in training set

train_img = []

train_lb = []

has_cactus = 0

no_cactus = 0

for i in range(len(cactus_label)):

    row = cactus_label.iloc[i]

    fileName = row['id'] 

    path = "../input/train/train/{}".format(fileName)

    im = mpimg.imread(path)

    if row['has_cactus'] == 1 and has_cactus < 4364:

        has_cactus+= 1

        train_lb.append(row['has_cactus'])

        train_img.append(im)

    elif row['has_cactus'] == 0 and no_cactus < 4364:

        no_cactus += 1

        train_lb.append(row['has_cactus'])

        train_img.append(im)





    

X_train, X_test, y_train, y_test = train_test_split(train_img, train_lb) 

X_train = np.array(X_train)

X_test = np.array(X_test)
import matplotlib.pyplot as plt

# Data to plot

labels = 'Has Cactus', 'No Cacuts'

sizes = [train_lb.count(1), train_lb.count(0)]

colors = ['yellowgreen', 'lightskyblue']

 

# Plot

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

start = time.time()



X_train_flatten = imageToFeatureVector(X_train)

X_test_flatten = imageToFeatureVector(X_test)



knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_flatten, y_train) 

score = knn.score(X_test_flatten, y_test)



end = time.time()

print("The run time of KNN is {:.3f} seconds".format(end-start))

print("KNN alogirthm's test score is: {:.3f}".format(score))
from sklearn.svm import LinearSVC



start = time.time()

linearKernel = LinearSVC().fit(X_train_flatten, y_train)

score = linearKernel.score(X_test_flatten,y_test)

end = time.time()



print("The run time of Linear SVC is {:.3f} seconds".format(end-start))

print("Linear SCV alogirthm's test score is: {:.3f}".format(score))
# try normalizing the features...

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

start = time.time()

scaler.fit(X_train_flatten)

X_test_normalized = scaler.transform(X_test_flatten)

X_train_normalized = scaler.transform(X_train_flatten)



linearKernel = LinearSVC().fit(X_train_normalized, y_train)

score = linearKernel.score(X_test_normalized,y_test)

end = time.time()

print("The run time of Linear SVC with normalized features is {:.3f} seconds".format(end-start))

print("Linear SCV with normalized features has test score of: {:.3f}".format(score))
import tensorflow as tf

start = time.time()

X_train_norm = tf.keras.utils.normalize(X_train, axis=1)

X_test_norm = tf.keras.utils.normalize(X_test, axis=1)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())



#add layers

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



#compile model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



#train 

model.fit(X_train_norm, np.array(y_train), epochs=10)



# Evaluate the model on test set

score = model.evaluate(X_test, np.array(y_test), verbose=0)

# Print test accuracy

print('\n', 'Test accuracy:', score[1])



end = time.time()

print("The run time of CNN is {:.3f} seconds".format(end-start))
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, DepthwiseConv2D, Flatten

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping



def create_model():

    model = Sequential()

        

    model.add(Conv2D(3, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))

    

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 32, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 64, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 128, kernel_size = 1, activation = 'relu'))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 256, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 2, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 512, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    model.add(DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'Same', use_bias = True))

    model.add(Conv2D(filters = 1024, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 2048, kernel_size = 1, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    

    model.add(Dense(470, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(128, activation = 'tanh'))



    model.add(Dense(1, activation = 'sigmoid'))



    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])

    

    return model
model = create_model()



history = model.fit(X_train, 

            np.array(y_train), 

            batch_size = 128, 

            epochs = 8, 

            validation_data = (X_test, np.array(y_test)),

            verbose = 1)



predictions = model.predict(X_test, verbose = 1)



# Evaluate the model on test set

score = model.evaluate(X_test, np.array(y_test), verbose=0)

# Print test accuracy

print('Test accuracy:', score[1])
scaler = preprocessing.StandardScaler()

scaler.fit(X_train_flatten)

X_test_normalized = scaler.transform(X_test_flatten)

X_train_normalized = scaler.transform(X_train_flatten)

test_flatten = imageToFeatureVector(test_img)

test_normalized = scaler.transform(test_flatten)

linearKernel = LinearSVC().fit(X_train_normalized, y_train)

predictions = linearKernel.predict(test_normalized)

sample['has_cactus'] = predictions

sample.head()
sample.to_csv('sub.csv', index= False)