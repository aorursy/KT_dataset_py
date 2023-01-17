#import the necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import zipfile as zf
import os
import csv
import gc
import operator
import random
from sklearn.cross_validation import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
from IPython.display import Image
from pathlib import Path

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K
from keras.models import Sequential
from keras import optimizers

#load the training data
trainData = pd.read_csv("../input/whale-categorization-playground/train.csv")

#See what is in the data
trainData.sample(5)

#trainData['Id'].value_counts()
#show sample image
Image(filename="../input/whale-categorization-playground/train/"+random.choice(trainData['Image'])) 
def prepareImages(data, m, dataset):
    
    print("Preparing images")
    
    X_train = np.zeros((m, 100, 100, 3))
    
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/whale-categorization-playground/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    count = 0
    
    print("Finished!")
            
    return X_train
#not used version of the image preparation class
#def prepareImages(data, m, dataset):
    
    #print("Preparing images")
    
    #X_train = np.zeros((m, 100, 100, 3))
    
    #count = 0
    
    #for fig in data['Image']:
        #file = Path("../input/whale-categorization-playground/"+dataset+"/"+fig)
        #if(file.is_file()):
            #img = image.load_img(file, target_size=(100, 100, 3))
        #else:
            #img = image.load_img("../input/augmented-data-whale/data/"+fig, target_size=(100, 100, 3))
        #x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #X_train[count] = x
        #if (count%500 == 0):
           # print("Processing image: ", count+1, ", ", fig)
        #count += 1
    
    #count = 0
    
   # print("Finished!")
            
    #return X_train
def prepareY(Y):

    values = array(Y)
    print(values.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    y = onehot_encoded
    print(y.shape)
    return y, label_encoder

#the next lines are used to test the code and do not need to run when using it
#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#print(inverted)
mod = Sequential()

mod.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

mod.add(BatchNormalization(axis = 3, name = 'bn0'))
mod.add(Activation('relu'))

mod.add(MaxPooling2D((2, 2), name='max_pool'))
mod.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
mod.add(Activation('relu'))
mod.add(AveragePooling2D((3, 3), name='avg_pool'))

mod.add(Flatten())
mod.add(Dense(500, activation="relu", name='rl'))
mod.add(Dropout(0.8))
mod.add(Dense(4251, activation='softmax', name='sm'))

print(mod.output_shape)

#opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
#I have used many variations of the CNN architecture and choice of optimizer.
#mod = Sequential()

#mod.add(Conv2D(32, (7, 7), strides = (1, 1), padding='same', name = 'conv0', input_shape = (100, 100, 3)))
#mod.add(BatchNormalization(axis = 3, name = 'bn0'))
#mod.add(Activation('relu'))
#print(mod.output_shape)

#mod.add(MaxPooling2D((2, 2), name='max_pool'))
#print(mod.output_shape)

#mod.add(Conv2D(64, (3, 3), strides = (1,1), padding='same', name="conv1"))
#mod.add(Activation('relu'))
#mod.add(BatchNormalization(axis = 3, name = 'bn1'))

#mod.add(AveragePooling2D((3, 3), name='avg_pool'))
#mod.add(AveragePooling2D((2, 2), name='avg_pool'))
#print(mod.output_shape)

#mod.add(Conv2D(128, (3, 3), strides = (1,1), padding='same', name="conv2"))
#mod.add(Activation('relu'))
#mod.add(BatchNormalization(axis = 3, name = 'bn2'))

#mod.add(MaxPooling2D((2, 2), name='max_pool2'))

#mod.add(Conv2D(128, (4, 4), strides = (1,1), padding='same', name="conv3"))
#mod.add(Activation('relu'))
#mod.add(BatchNormalization(axis = 3, name = 'bn3'))

#mod.add(AveragePooling2D((2, 2), name='avg_pool1'))
#print(mod.output_shape)

#mod.add(Flatten())
#print(mod.output_shape)

#mod.add(Dense(500, activation="relu", name='rl'))
#mod.add(Dropout(0.6))

#mod.add(Dense(500, activation="relu", name='r2'))
#mod.add(Dropout(0.8))

#mod.add(Dense(4251, activation='softmax', name='sm'))

#print(mod.output_shape)

#opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
X = prepareImages(trainData, 9850, "train")

#put all the values of the training data in the range between 0 and 1
X /= 255

print("Shape X-train: ", X.shape)
Y = trainData['Id']

print("Shape Y-train: ", Y.shape)

#The next lines are used for testing - not necessary for the code
#labels = trainData['Id'].unique()
#print("Labels in data set: ", labels.shape)
#labelsTrain = Y.unique()
#print("Labels in training set: ", labelsTrain.shape)

y, label_encoder = prepareY(Y)
#used to prepare augmented data - not used here
#train2 = []

#for i in trainData['Image']:
    #temp = i.split('.')
    #temp[0] = temp[0]+'f.jpg'
    #train2.append(temp[0])

#col = ['Image']
#train2 = pd.DataFrame(train2, columns=col)

#train3 = []

#for i in trainData['Image']:
    #temp = i.split('.')
    #temp[0] = temp[0]+'r.jpg'
    #train3.append(temp[0])
    
#train3 = pd.DataFrame(train3, columns=col)

#train4 = []

#for i in trainData['Image']:
    #temp = i.split('.')
    #temp[0] = temp[0]+'r2.jpg'
    #train4.append(temp[0])
    
#train4 = pd.DataFrame(train4, columns=col)

#imageData = np.concatenate((np.reshape(trainData['Image'], (9850, 1)), train2), axis=0)
#imageData = np.concatenate((imageData, train3), axis=0)
#imageData = np.concatenate((imageData, train4), axis=0)
#idData = np.concatenate((trainData['Id'], trainData['Id']), axis=0)
#idData = np.concatenate((idData, trainData['Id']), axis=0)
#idData = np.concatenate((idData, trainData['Id']), axis=0)
#tempData = np.concatenate((imageData, np.reshape(idData, (39400, 1))), axis=1)

#cols = ['Image', 'Id']
#data = pd.DataFrame(tempData, columns=cols)

#print(data.shape)
history = mod.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()
#training with augmented data
#for i in range(0, 70000):
    #batch = data.sample(32)
    #Ydata = np.zeros((32, 4251))
    #X = prepareImages(batch, 32, "train")
    #X /= 255
    #ytemp = label_encoder.transform(batch['Id'])
    #for j in range(0,32):
        #Ydata[j][ytemp[j]] = 1
    
    #history = mod.train_on_batch(X, Ydata)

#print(mod.metrics_names)
#print(history)
#continued training
#for i in range(0, 197):
    #batch = data[i*50: i*50+50]
    #Ydata = np.zeros((50, 4251))
    #X = prepareImages(batch, 50, "train")
    #X /= 255
    #ytemp = label_encoder.transform(batch['Id'])
    #for j in range(0,50):
        #Ydata[j][ytemp[j]] = 1
    
    #history = mod.train_on_batch(X, Ydata)
    
#print(mod.metrics_names)
#print(history)
#another version of the code to train the CNN
#gc.collect() 
#Y = trainData['Id']
#X = prepareImages(trainData, 9850, "whale-categorization-playground/train")
#y, label_encoder = prepareY(Y)

#X /= 255

#train2 = []

#for i in trainData['Image']:
    #temp = i.split('.')
    #temp[0] = temp[0]+'f.jpg'
    #train2.append(temp[0])

#col = ['Image']
#train2 = pd.DataFrame(train2, columns=col)

#gc.collect()
#X2 = prepareImages(train2, 9850, "humpback-whale-flipped/train2")
#gc.collect()

#X2 /= 255

#X = np.concatenate((X, X2), axis=0)
#y = np.concatenate((y, y), axis=0)

#print(X.shape)
#print(y.shape)
#gc.collect() 
#another version of the code to train the CNN, here using Keras for data augmentation
#gc.collect()
#datagen = image.ImageDataGenerator( 
    #featurewise_center=True, 
    #featurewise_std_normalization=True, 
    #rotation_range=20, 
    #width_shift_range=0.2, 
    #height_shift_range=0.2, 
    #horizontal_flip=True)

#datagen.fit(X[0:6000])

#gc.collect()
#mod.fit_generator(datagen.flow(X, y, batch_size=100), steps_per_epoch=len(X) / 100, epochs=30, initial_epoch=0)
#history = mod.fit(X, y, epochs=2, batch_size=100, verbose=1, shuffle=True)
#gc.collect()
#plot how the accuracy changes as the model was trained
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
#open test data
test = os.listdir("../input/whale-categorization-playground/test/")
print(len(test))

#separate data into different DataFrames due to memory constraints
col = ['Image']
testData1 = pd.DataFrame(test[0:3899], columns=col)
testData2 = pd.DataFrame(test[3900:7799], columns=col)
testData3 = pd.DataFrame(test[7800:11699], columns=col)
testData4 = pd.DataFrame(test[11700:15609], columns=col)
testData = pd.DataFrame(test, columns=col)
#X_test = prepareImages(testData1, 15610, "test")
gc.collect()
X = prepareImages(testData1, 3900, "test")
X /= 255
predictions1 = mod.predict(np.array(X), verbose=1)
gc.collect()
X = prepareImages(testData2, 3900, "test")
X /= 255
predictions2 = mod.predict(np.array(X), verbose=1)
gc.collect()
X = prepareImages(testData3, 3900, "test")
X /= 255
predictions3 = mod.predict(np.array(X), verbose=1)
gc.collect()
X = prepareImages(testData4, 3910, "test")
X /= 255
predictions4 = mod.predict(np.array(X), verbose=1)
gc.collect()
#concatenate all the predictions in the same vector
predictions = np.concatenate((predictions1, predictions2), axis=0)
predictions = np.concatenate((predictions, predictions3), axis=0)
predictions = np.concatenate((predictions, predictions4), axis=0)
gc.collect()
print(predictions.shape)
print(predictions)
#choose predictions with highest probability. For each value I choose, I set the probability to zero, so it can't be picked again.
print(predictions.shape)

copy_pred = np.copy(predictions)
idx = np.argmax(copy_pred, axis=1)
copy_pred[:,idx] = 0
idx2 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx2] = 0
idx3 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx3] = 0
idx4 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx4] = 0
idx5 = np.argmax(copy_pred, axis=1)
#convert the one-hot vectors to their names
results = []

print(idx[0:10])
print(idx2[0:10])
print(idx3[0:10])
print(idx4[0:10])
print(idx5[0:10])
threshold = 0.05 #threshold - only consider answers with a probability higher than it
for i in range(0, predictions.shape[0]):
#for i in range(0, 10):
    each = np.zeros((4251, 1))
    each2 = np.zeros((4251, 1))
    each3 = np.zeros((4251, 1))
    each4 = np.zeros((4251, 1))
    each5 = np.zeros((4251, 1))
    if((predictions[i, idx5[i]] > threshold)):
        each5[idx5[i]] = 1
        each4[idx4[i]] = 1
        each3[idx3[i]] = 1
        each2[idx2[i]] = 1
        each[idx[i]] = 1
        tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0], label_encoder.inverse_transform([argmax(each5)])[0]]
    else:
        if((predictions[i, idx4[i]] > threshold)):
            print(predictions[i, idx4[i]])
            each4[idx4[i]] = 1
            each3[idx3[i]] = 1
            each2[idx2[i]] = 1
            each[idx[i]] = 1
            tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0]]
        else:
            if((predictions[i, idx3[i]] > threshold)):
                each3[idx3[i]] = 1
                each2[idx2[i]] = 1
                each[idx[i]] = 1
                tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0]]
            else:
                if((predictions[i, idx2[i]] > threshold)):
                    each2[idx2[i]] = 1
                    each[idx[i]] = 1
                    tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0]]
                else:
                    each[idx[i]] = 1
                    tags = label_encoder.inverse_transform([argmax(each)])[0]
    results.append(tags)
#write the predictions in a file to be submitted in the competition.
myfile = open('output.csv','w')

column= ['Image', 'Id']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)

for i in range(0, testData.shape[0]):
    pred = ""
    if(len(results[i])==5):
        if (results[i][4]!=results[i][0]):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3] + " " + results[i][4]
        else:
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
    else:
        if(len(results[i])==4):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
        else:
            if(len(results[i])==3):
                pred = results[i][0] + " " + results[i][1] + " " + results[i][2]
            else:
                if(len(results[i])==2):
                    pred = results[i][0] + " " + results[i][1]
                else:
                    pred = results[i]
            
    result = [testData['Image'][i], pred]
    #print(result)
    wrtr.writerow(result)
    
myfile.close()
