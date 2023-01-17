# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import PIL

from sklearn.model_selection import train_test_split

import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, BatchNormalization, Convolution2D , MaxPooling2D

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

import itertools

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical

from keras.preprocessing.image import load_img, img_to_array
#Among all the content we got from the input folder we choose to keep only the train and test folders . 

#The other files will be used later on

fichier=os.listdir("../input")

fichier.remove('train.csv')

fichier.remove('sample_submission.csv')

fichier  
#The folders train  and test countain images refered to by their 'id' followed by the type of the file (jpg).

imgtype = '/*.jpg'

train = sorted([x for x in os.listdir("../input" + '/'+ 'train' + '/train'  ) if x.endswith(imgtype[2:])])

test = sorted([x for x in os.listdir("../input" + '/'+ 'test' + '/test') if x.endswith(imgtype[2:])])

train
#We open the CSV file which countains two columns: 'id'and 'has cactus'

csv_train = pd.read_csv("../input/" + 'train.csv')

sns.countplot('has_cactus', data=csv_train)

plt.title('Classes', fontsize=15)

plt.show()

csv_train.has_cactus=csv_train.has_cactus.astype(str) #This variable is turned to string to match the data format 

                                                      #required by the augmentation function we'll use

#The function ImageDataGenerator allows here to multiply the data by the value provided.

#Batch Size corresponds to the number of samples that will be passed through our network at one time. 

datagen=ImageDataGenerator(rescale=1./255)

batch_size=140 #so that the number of iterations will be 100 since the training sample countains 14000 images
#--------The Data Augmentation

#The initial train sample is split between Training sample and validation sample following the rule of 80%/20%

#The function flow_from_dataframe will provide a new sample with images randomly changed following the parameters fixed



train_generator=datagen.flow_from_dataframe(dataframe=csv_train[:14001],directory="../input" + '/'+ 'train' + '/train',x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))

validation_generator=datagen.flow_from_dataframe(dataframe=csv_train[14000:],directory="../input" + '/'+ 'train' + '/train',x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))

y_train=csv_train[:14001]

y_val=csv_train[14000:]
nb_train_samples = 14001

nb_validation_samples = 3500
#--------Convolutionnal Neural Network model

#The model we chose consists in three steps each time: convolution, pooling, and dropout.

#1-Convolution2D function consists in applying a Convolutional kernel, of size 3x3. 

#2-MaxPooling uses a kernel 2x2 to downsize the information countained in the image and keep only the pixel with max value

#3-Dropout

model = Sequential() #Creation of an empty neural network

model.add(Convolution2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu')) #followed by an activation layer

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.1))



model.add(Convolution2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.1))



model.add(Convolution2D(128, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.1))



#Flatten 3d feature maps to 1D

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))



#compile the model

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#generate a summary of the model

model.summary()
#Stops the fit generator when the value of the accuaracy stagnates

callbacks = [EarlyStopping(monitor='val_loss', patience=4)]
#--------Learning for the training set

history = model.fit_generator(

    train_generator, 

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=20,

    callbacks=callbacks,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size,

)
# plot training history

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.plot(history.history['val_acc'], label='acc_test')

plt.plot(history.history['acc'], label='acc_train')

plt.legend()

plt.show()

#the accuaracy of the model is 98%.
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
Y_pred = model.predict_classes(validation_generator)

confusion_mtx = confusion_matrix(validation_generator.classes, Y_pred) 

plot_confusion_matrix(confusion_mtx, classes = range(2))



#Its seems that the CNN built recognizes well when an image contains a Cactus, 

#but the  big number of real '1's is because they represented a biggest share of the training set.
datagen1=ImageDataGenerator(rotation_range=30, width_shift_range=0.2, 

                             height_shift_range=0.2, zoom_range=0.2, 

                             horizontal_flip=True, vertical_flip=True, 

                             validation_split=0.1,rescale=1./255)

batch_size=140
train_generator1=datagen1.flow_from_dataframe(dataframe=csv_train[:14001],directory="../input" + '/'+ 'train' + '/train',x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator1=datagen.flow_from_dataframe(dataframe=csv_train[14000:],directory="../input" + '/'+ 'train' + '/train',x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))



y_train=csv_train[:14001]

y_val=csv_train[14000:]
model1 = Sequential()

model1.add(Convolution2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

model1.add(MaxPooling2D(pool_size = (2, 2)))

model1.add(Dropout(rate = 0.1))

model1.add(Convolution2D(64, (3, 3), activation = 'relu'))

model1.add(MaxPooling2D(pool_size = (2, 2)))

model1.add(Dropout(rate = 0.1))

#flatten 3d feature maps to 1D

model1.add(Flatten())

model1.add(Dense(64, activation = 'relu'))

model1.add(Dense(1, activation = 'sigmoid'))



#compile the model

model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



model1.summary()



callbacks = [EarlyStopping(monitor='val_loss', patience=4)]



history1 = model1.fit_generator(

    train_generator1, 

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=20,

    callbacks=callbacks,

    validation_data=validation_generator1,

    validation_steps=nb_validation_samples // batch_size,

)
Y_pred1 = model1.predict_classes(train_generator)

confusion_mtx = confusion_matrix(train_generator.classes, Y_pred1) 

plot_confusion_matrix(confusion_mtx, classes = range(2))
#Images of the test folder are loaded into X_tst and resized so they can  match the format for which we did the training

X_tst = []

Test_imgs = []

for img_id in os.listdir("../input" + '/'+ 'test' + '/test'):

    img = cv2.imread("../input" + '/'+ 'test' +'/' +'test' +'/' + img_id)

    img2 = cv2.resize(img, (150,150))

    X_tst.append(img2)     

    

    Test_imgs.append(img_id)

X_tst = np.asarray(X_tst)

X_tst = X_tst.astype('float32')

X_tst /= 255

#We apply the final weights on the test sample... 

test_predictions = model.predict_classes(X_tst)

test_predictions
#... and save the result in csv file where the first column is the 'id' and the second is the prediction

sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])

sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

sub_df['id'] = ''

cols = sub_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub_df=sub_df[cols]

for i, img in enumerate(Test_imgs):

    sub_df.set_value(i,'id',img)
sub_df.head()
sub_df.to_csv('submission.csv',index=False)