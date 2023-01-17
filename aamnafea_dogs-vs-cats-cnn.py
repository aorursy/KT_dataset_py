#import libraries
import pandas as pd
import numpy as np
import os, random ,cv2
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



#specify train and test datasets paths
train_path = '../input/train/'
test_path = '../input/test/'
#define image dimensions 
rows = 150
cols = 150
channels = 3
#create a list of train image paths "including image name"
train_images = [train_path+i for i in os.listdir(train_path)]
train_dogs = [train_path+i for i in os.listdir(train_path) if 'dog' in i]
train_cats = [train_path+i for i in os.listdir(train_path) if 'cat' in i]
#create a list of test image paths "including image name"
test_images = [test_path+i for i in os.listdir(test_path)]
train_images = train_dogs[:3000] + train_cats[:3000]
#randomly shuffle train images
random.shuffle(train_images)
def prep_data(image_path_list):
    x=[]
    y=[]
    for i in image_path_list:
        x.append(cv2.resize(plt.imread(i), #read then resize image 
        (rows,cols), interpolation=cv2.INTER_CUBIC))  #appened new image to x
    for j in image_path_list: #create a label and append it to y 
        if 'dog' in j:
            y.append(1)
        elif 'cat' in j:
            y.append(0)
    return x,y        
X, y = prep_data(train_images)
#split X,y into a train and validation data sets
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=(1/3), random_state=1)
X_test, y_test = prep_data(test_images)
#create a keras CNN model from sctarch
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(rows, cols, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='RMSprop', metrics=['accuracy'], loss='binary_crossentropy')
model.summary()
#create a data generator object with some image augmentation specs
datagen = ImageDataGenerator(
    rescale=1./ 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

#create an iterator for both train and valid sets
train_gen = datagen.flow(x=np.array(X_train), y=y_train, batch_size=50)
valid_gen = datagen.flow(x=np.array(X_val), y=y_val, batch_size=50)
#train/validate model
model.fit_generator(train_gen, steps_per_epoch=40, epochs=50, verbose=1, validation_data=valid_gen, validation_steps=20)
#create a data generator object for testing
datagen = ImageDataGenerator(rescale = 1./255)
test_gen = datagen.flow(np.array(X_test), batch_size = 100)
#predict
predictions = model.predict_generator(test_gen, steps=125, verbose=1)
predictions_dogs=[]
for i in predictions:
    predictions_dogs.append(i[0])
#submit
id_num = range(1, len(predictions_dogs) + 1)
submission = pd.DataFrame({"id": id_num, "label":predictions_dogs})
submission.to_csv("submission.csv", index = False)