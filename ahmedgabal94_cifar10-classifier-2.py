# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential,Model, load_model
from keras import losses
from keras import initializers
from keras.layers import Activation ,Dropout ,Flatten,Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D ,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,rmsprop
from keras import regularizers 
# from keras.callbacks import ModelCheckpoint ,History 
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import data

csv_test = pd.read_csv('../input/sample_submission.csv')
# read test CSV 
csv_test.head(10)
# Generate Labels
csv_train = pd.read_csv('../input/train_labels.csv')
# read training CSV
csv_train.head(10)
labels = pd.Series(csv_train['Category'])
#print(targets_series)
# load train , test 
x_train = np.load('../input/train_images.npy')
x_test = np.load('../input/test_images.npy')
# scalling 

y_train = labels
lb= LabelBinarizer()
y_train=lb.fit_transform(y_train) 
x_train = x_train /255
x_test = x_test/255
print('labels',y_train.shape) #after one hot so the shape is 50000*10 "10"each classes 0000100000 
print('x_train',x_train.shape)
print('x_test',x_test.shape)
print ('csv_text',csv_test.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.2, random_state=42,
                                                      stratify=y_train)
#VGG
np.random.seed(0)
weight_decay = 0.0005
def creat_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same',data_format="channels_first", input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3), padding='same',data_format="channels_first",))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.7))
    
    model.add(Conv2D(256, (3, 3), padding='same',data_format="channels_first",kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l2(weight_decay))) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3),data_format="channels_first",kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7)) 

    model.add(Flatten())
    
    model.add(Dense(num_classes, activation='softmax'))
    opt = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999) # try momentum 
    #opt=rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics =['accuracy'])
    model.summary()
    return model 
batch_size = 128
num_classes = 10 
epochs =250

# Augmentation Data 
# def create_datagen(X_train):
#     data_generator = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     )
#     data_generator.fit(X_train)
#     return data_generator

# data_generator=create_datagen(X_train) 

# checkpoint_path= 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
# #Create checkpoint callback
# checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_loss',verbose=0,save_best_only=True,
#                             save_weights_only=True,
#                             mode='min') 
model =creat_model() 

# cnn=model.fit_generator(data_generator.flow(X_train,Y_train, batch_size=batch_size),
#                     steps_per_epoch=len(X_train) // batch_size,
#                     epochs=epochs,
#                     validation_data=(X_valid,Y_valid),
#                     verbose=1)

cnn=model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs, validation_data=(X_valid,Y_valid),shuffle=True)

# Plots for training and testing process: loss and accuracy
 
plt.figure(0)
plt.plot(cnn.history['acc'],'b')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
 
 
plt.figure(1)
plt.plot(cnn.history['loss'],'b')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
 
 
plt.show()
prediction =model.predict(x_test,verbose=1) 
indx= np.argmax(prediction ,axis=1)
label= lb.classes_[indx]
label
# creat the submition file 

data={'ID': np.arange(1,label.shape[0]+1),
       'Category': label}
predict= pd.DataFrame(data)
predict=predict[['ID','Category']]
predict.to_csv('submission.csv', index=False)

# classes = csv_test.columns.values[1:] 
# frame = pd.DataFrame(prediction,index=range(1,200001), columns=classes)
# frame.to_csv("predicted_test.csv", index_label='Id')


