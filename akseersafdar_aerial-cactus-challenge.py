#making the imports



import pandas as pd

import numpy as np



import cv2

import os

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from keras.models import Sequential

from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

#define the directory



train_directory="../input/train/train"

test_directory="../input/test/test"

train=pd.read_csv('../input/train.csv')
#checking the head

train.head()
plt.figure(figsize = (8,6))

sns.set_style('dark')

sns.countplot(train['has_cactus'])

plt.show()
#lets display a random image from the training set



img = cv2.imread('../input/train/train/0148bb4a295cf49c0169d69a4a63df7e.jpg')

plt.figure(figsize = (10,8))

plt.imshow(img)

plt.xticks([])

plt.yticks([])

plt.show()
#lets display a random image from the training set



img = cv2.imread('../input/train/train/0de4702853bd3667fb24db3a8dcc07bd.jpg')

plt.figure(figsize = (10,8))

plt.imshow(img)

plt.xticks([])

plt.yticks([])

plt.show()
#lets check the shape of image

img.shape
#define the parameters for ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255, validation_split= 0.2, shear_range= 0.2, 

                                  zoom_range= 0.2, horizontal_flip= True)
#converting the has_cactus to a string

train['has_cactus'] = train['has_cactus'].astype(str)
#lets use data generator to make training and validation set.



train_generator = train_datagen.flow_from_dataframe(train, 

                                                    directory= train_directory, 

                                                    subset= 'training',

                                                    x_col= 'id',

                                                    y_col= 'has_cactus',

                                                    target_size= (32,32),

                                                    class_mode= 'binary'

                                                   )





test_generator = train_datagen.flow_from_dataframe(train,

                                                  directory= train_directory,

                                                  subset= 'validation',

                                                  x_col= 'id',

                                                  y_col= 'has_cactus',

                                                  target_size= (32,32),

                                                  class_mode= 'binary'

                                                  )
#define the model layers

model = Sequential()



model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))

model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))



model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))



model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(MaxPool2D(2,2))



model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation = 'sigmoid'))

#lets compile the model



model.compile(loss = 'binary_crossentropy',

             optimizer= Adam(),

             metrics= ['accuracy'])
#model summary

model.summary()
#lets train the model for 20 epochs



history = model.fit_generator(train_generator,

                             steps_per_epoch= 2000, 

                             epochs= 20, 

                             validation_data= test_generator,

                             validation_steps= 64)
#convert the results to a data frame

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist
#plotting the results to see difference between train and validation accuracy/loss



plt.figure(figsize = (8,6))

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.plot(hist['epoch'],hist['val_acc'], label = 'Val Accuracy')

plt.plot(hist['epoch'],hist['acc'], label = 'Train Accuracy')

plt.xticks(range(0,20))

plt.legend(loc = 'lower right')

plt.title('Accuracy')

plt.show()



plt.figure(figsize = (8,6))

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.plot(hist['epoch'],hist['val_loss'], label = 'Val Loss')

plt.plot(hist['epoch'],hist['loss'], label = 'Train Loss')

plt.xticks(range(0,20))

plt.legend()

plt.title('Loss')

plt.show()
#getting the test set ready to make predictions



ids = []

X_test = []



for image in os.listdir(test_directory):

    

    ids.append(image.split('.')[0])

    path = os.path.join(test_directory, image)

    X_test.append(cv2.imread(path))

    

X_test = np.array(X_test)

X_test = X_test.astype('float32')/ 255
#making the predictions

predictions = model.predict(X_test)
#writing to submission file



my_sub = pd.read_csv('../input/sample_submission.csv')

my_sub['id'] = ids

my_sub['has_cactus'] = predictions
#convert the probability to 0s and ones. 

def cvt_prob(x):

    

    if x >= 0.5:

        return 1

    else:

        return 0

    

my_sub['has_cactus'] = my_sub['has_cactus'].apply(cvt_prob)    
#see the count of 0s and 1s 

plt.figure(figsize = (8,6))

sns.set_style('dark')

sns.countplot(my_sub['has_cactus'])

plt.show()
#write to submission file

my_sub.to_csv('my_sub1.csv',index= False)