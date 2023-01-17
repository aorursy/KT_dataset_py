# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
%matplotlib inline

import os
import random
import gc
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

train_dir = '../input/train'
test_dir = '../input/test'

train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i ]  #get dog images
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i ]  #get cat images

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]  #get test images

train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
random.shuffle(train_imgs)  # shuffle it randomly


#Clear list that are useless
del train_dogs
del train_cats

gc.collect()  #collect garbage to save memory


#Lets declare our image dimensions
#we are using coloured images.
nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image

def read_and_process_image(list_of_images):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  # Read the image
        #get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
            
    return X,y


X,y = read_and_process_image(train_imgs)
import matplotlib.image as mping
for ima in train_imgs[0:3]:
    img = mping.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()
    
X[0]
y
plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5/columns+1,columns,i+1)
    plt.imshow(X[i])
    
import seaborn as sns
del train_imgs
gc.collect()

#Convert list to numpy array
X = np.array(X)
y = np.array(y)

#Lets plot the label to be sure we just have two class
sns.countplot(y)
plt.title('Labels for Cats and Dogs')

print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)
# Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=2)

print("Shape of train images is:",X_train.shape)
print("Shape of validation images is:",X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:",y_val.shape)

# clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note:batch size should be a factor of 2.***4,8,16,32,64...***
batch_size=32
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid')) #Sigmoid function at the end because we have just two classes

#Lets see our model
model.summary()
#We'll use the RMSprop optimizer with a learing rate of 0.0001
#We'll use binary_crossentropy loss because its a binary classification
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#Lets create the alugmentation configuration
#This helps prevent overfitting,since we are using a small dataset
train_datagen = ImageDataGenerator(rescale=1./255, #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)
val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale

#Create the image generators
train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size)

#The training part
#We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                             steps_per_epoch=ntrain//batch_size,
                             epochs=64,
                             validation_data=val_generator,
                             validation_steps=nval//batch_size)
#Save the model
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
#lets plot the train and val curve
#get the details form the history object
acc=history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

#train and validation accuracy
plt.plot(epochs,acc,'b',label='Training accurarcy')
plt.plot(epochs,val_acc,'r',label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Trailn and validation loss
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

#Now lets predict on the first 10 Images of the test set
X_test,y_test = read_and_process_image(test_imgs[0:10]) #Y_test in this case will be empty
x=np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i=0
text_labels =[]
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x,batch_size=1):
    pred =model.predict(batch)
    if pred>0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(5/columns+1,columns,i+1)
    plt.title('This is a '+text_labels[i])
    imgplot=plt.imshow(batch[0])
    i+=1
    if i%10==0:
        break
plt.show()