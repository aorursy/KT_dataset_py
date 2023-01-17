import numpy as np
import pandas as pd
from PIL import Image
import os
import math
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline
'''Pulling the photos from folders with their paths'''

path_0 = []
train_path_0 = "../input/hand-sign-language-digit-dataset-for-0-5/0/"                #zero
for path in os.listdir(train_path_0):
    if '.JPG' in path:
        path_0.append(os.path.join(train_path_0, path))
        
path_1 = []
train_path_1 = "../input/hand-sign-language-digit-dataset-for-0-5/1/"                #one
for path in os.listdir(train_path_1):
    if '.JPG' in path:
        path_1.append(os.path.join(train_path_1, path))
        
path_2 = []
train_path_2 = "../input/hand-sign-language-digit-dataset-for-0-5/2/"                #two
for path in os.listdir(train_path_2):
    if '.JPG' in path:
        path_2.append(os.path.join(train_path_2, path))

path_3 = []
train_path_3 = "../input/hand-sign-language-digit-dataset-for-0-5/3/"                #three
for path in os.listdir(train_path_3):
    if '.JPG' in path:
        path_3.append(os.path.join(train_path_3, path))
        
path_4 = []
train_path_4 = "../input/hand-sign-language-digit-dataset-for-0-5/4/"                #four
for path in os.listdir(train_path_4):
    if '.JPG' in path:
         path_4.append(os.path.join(train_path_4, path))
        
path_5 = []
train_path_5 = "../input/hand-sign-language-digit-dataset-for-0-5/5/"                #five
for path in os.listdir(train_path_5):
    if '.JPG' in path:
        path_5.append(os.path.join(train_path_5, path))

print("Number of pics for each digit:")
print((len(path_0), len(path_1), len(path_2), len(path_3), len(path_4), len(path_5)))

print("Total pics in the dataset: " + str(len(path_0) + len(path_1) + len(path_2) + len(path_3) + len(path_4) + len(path_5)))
'''Load training set'''

'''total pics in training set =  1237
    training_set = 1230 --- 205 for each digit'''

train_set_orig = np.zeros((1230, 64, 64, 3), dtype='float32')

for i in range(205):
    image = Image.open(path_0[i])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
    
for i in range(205, 410):                                                           #loading "one"
    image = Image.open(path_1[i - 205])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
        
for i in range(410, 615):                                                           #loading "two"
    image = Image.open(path_2[i - 410])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
        
for i in range(615, 820):                                                           #loading "three"
    image = Image.open(path_3[i - 615])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
    
for i in range(820, 1025):                                                           #loading "four"
    image = Image.open(path_4[i - 820])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
        
for i in range(1025, 1230):                                                          #loading "five"
    image = Image.open(path_5[i - 1025])
    img_resized = image.resize((64,64))
    train_set_orig[i] = np.asarray(img_resized)
'''Labelling the training set having 6-dimensional vector of o's and 1's with 1 where index  = digit and zero otherwise'''

train_y_ = np.zeros((1230, 6))

for i in range(205):                                                               #labelling "zero"
    train_y_[i, 0] = 1

for i in range(205, 410):                                                          #labelling "one"
    train_y_[i, 1] = 1
        
for i in range(410, 615):                                                          #labelling "two"
    train_y_[i, 2] = 1
    
for i in range(615, 820):                                                          #labelling "three"
    train_y_[i, 3] = 1
    
for i in range(820, 1025):                                                          #labelling "four"
    train_y_[i, 4] = 1
    
for i in range(1025, 1230):                                                         #labelling "five"
    train_y_[i, 5] = 1

m_train = train_set_orig.shape[0]
num_px = train_set_orig.shape[1]

print("SUMMARY OF DATASET:")
print ("Number of training examples: m_train = " + str(m_train))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_orig.shape))
print ("train_set_y shape: " + str(train_y_.shape))
''''Suffling training set pics'''

np.random.seed(0)
m_train = train_set_orig.shape[0]
permutation = list(np.random.permutation(m_train))
train_set_x = train_set_orig[permutation, :]
train_y = train_y_[permutation, :]
train_x = train_set_x.reshape(1230,-1)
print ("train_set_x_flatten shape: " + str(train_x.shape))
print ("train_set_y shape: " + str(train_y.shape))
'''Standardizing dataset'''

train_x = train_x /255
'''Example of an image'''

index = 20
plt.imshow(np.uint8(train_set_x[index]), interpolation='nearest')
plt.show()
print(np.where(train_y[index] == 1)[0])
'''Making sequential deep learning model using keras
   input layer: shape -- ((64, 64, 3), number of examples)
   layer 1: -- Conv2D with 50 filters (3, 3) and stride of 1, with "relu" activation, + max pooling with (2, 2)
   layer 2: -- Conv2D with 75 filters (3, 3) and stride of 1, with "relu" activation, + max pooling with (2, 2)
   layer 3: -- Conv2D with 100 filters (3, 3) and stride of 1, with "relu" activation,+ max pooling with (2, 2)
   layer 4: -- Fully connected with 1024 units size with relu activation
   layer 5: -- Fully connected with 1024 units size with relu activation
   layer 6: -- Fully connected with 6 units size with softmax activation (output layer)'''

model = Sequential()

model.add(layers.Conv2D(input_shape = (64, 64, 3), filters = 50, strides = 1,
                        kernel_size = (3, 3), padding = "same", activation = "relu"))   #layer 1
model.add(layers.MaxPooling2D(pool_size = (2, 2), padding = "same"))

model.add(layers.Conv2D(filters = 75, strides = 1,
                        kernel_size = (3, 3), padding = "same", activation = "relu"))   #layer 2
model.add(layers.MaxPooling2D(pool_size = (2, 2), padding = "same"))

model.add(layers.Conv2D(filters = 100, strides = 1,
                        kernel_size = (3, 3), padding = "same", activation = "relu"))   #layer 3
model.add(layers.MaxPooling2D(pool_size = (2, 2), padding = "same"))

model.add(layers.Flatten())
model.add(Dense(units = 1024, activation = "relu"))                                      #layer 4
model.add(layers.Dropout(0.2))
model.add(Dense(units = 1024, activation = "relu"))                                      #layer 5
model.add(layers.Dropout(0.2))
model.add(Dense(units = 6, activation = "softmax"))                                      #output layer

'''using "Adam" optimizer'''

opt = keras.optimizers.Adam(learning_rate = 0.0001)

'''Compiling the model using the "categorical_crossentropy" loss function'''

model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])

model.summary()
'''Fitting the model using training set'''

language = model.fit(train_set_x, train_y, epochs = 20, batch_size = 32, validation_split = 0.05)
print("Training set accurarcy: " + str((language.history["accuracy"])[-1]*100) + "%")
print("Validation set accurarcy: " + str((language.history["val_accuracy"])[-1]*100) + "%")
plt.figure(figsize=(15, 5))

plt.subplot(1,2,1)

plt.plot(language.history["accuracy"], label = "training set")
plt.plot(language.history["val_accuracy"], label = "validation set")
plt.title("accuracy versus epochs curve")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')

plt.subplot(1,2,2)

plt.plot(language.history["loss"], label = "training set")
plt.plot(language.history["val_loss"], label = "validation set")
plt.title("loss versus epochs curve")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')

plt.show()