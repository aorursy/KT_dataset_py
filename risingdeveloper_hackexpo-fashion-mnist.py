import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import random
import gc   #Gabage collector for cleaning deleted data from memory

print(os.listdir("../input"))

#Create the training dataset
train = []
test = []
for j in range(10):
    img_list = ['../input/train/train/' + str(j) + '/{}'.format(i) for i in os.listdir('../input/train/train/{}'.format(j))]
    train = train + img_list

test = ['../input/test/test/{}'.format(i) for i in os.listdir('../input/test/test/')]
#Create the labels column
labels = []
for i in range(10):
    for j in range(6000):
        labels.append(i)
    
#Lets declare our image dimensions
#we are using coloured images. 
nrows = 100
ncolumns = 100
channels = 1  #change to 1 if you want to use grayscale image

#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images, labels):
    """
    Returns 2 arrays: 
        X is an array of images
        Y is the labels
    """
    X = [] # images
    
    for image in list_of_images:
#       X.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        
    #Convert to numpy array
    X = np.array(X)
    if labels == None:
        return X
    else:
        y = np.array(labels)
        return (X,y)
    
    
X,y = read_and_process_image(train, labels)
print("Shape of training data:", str(X.shape))
print("Shape of labels:", str(y.shape))
#Randomly shuffle both arrays
randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
y = y[randomize]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#let's display the first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y[i]])
#Lets plot the label to be sure we have 10 classes
sns.countplot(y)
plt.title('Labels for Fashion Mnist')

#Lets split the data into train and test set
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

y_cat = to_categorical(y)
# X = X.reshape(len(X), 150, 150, 3)
# X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.20, random_state=2)
ntrain = len(X)
# nval = len(X_val)
num_classes = 10
batch_size = 32
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(100,100,3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))      
fashion_model.add(Dropout(0.2))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.summary()
from keras import optimizers
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
#This helps prevent overfitting, since we are using a small dataset
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale
#Create the image generators
train_generator = train_datagen.flow(X, y_cat,batch_size=batch_size)
# val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
#The training part
#We train for 64 epochs with about 100 steps per epoch
history = fashion_model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=40)
#                               validation_data=val_generator,
#                               validation_steps=nval // batch_size)
#Save the model
fashion_model.save_weights('model_wieghts.h5')
fashion_model.save('model_keras.h5')
# #get the details form the history object
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# #Train and validation accuracy
# plt.plot(epochs, acc, 'b', label='Training accurarcy')
# plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
# plt.title('Training and Validation accurarcy')
# plt.legend()

# plt.figure()
# #Train and validation loss
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()

# plt.show()
sample_sub = pd.read_csv('../input/sample-submission.csv')
sample_sub.head()
pred_test = read_and_process_image(test, None)
#Scale test set
pred_test = pred_test * 1./255
predicted_classes = fashion_model.predict_classes(pred_test)
imageId = []
for i,id in enumerate(test):
    imageId.append(id.split('/')[4])
    
sample_sub['Category'] = predicted_classes
sample_sub['ImageID'] = imageId
sample_sub.head()
sample_sub.to_csv('submission1.csv', index=False)
