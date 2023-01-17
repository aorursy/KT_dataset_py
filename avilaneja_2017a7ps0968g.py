import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

filenames = os.listdir("../input/nnfl-lab-1/training/training/")
categories = []
filenames_new = []
for filename in filenames:
    category = filename.split('_')[0]
    if(category == 'chair'):
        categories.append(0)
        filenames_new.append(filename)
    elif category == 'kitchen':
        categories.append(1)
        filenames_new.append(filename)
    elif category == 'knife' :
        categories.append(2)
        filenames_new.append(filename)
    elif category == 'saucepan' :
        categories.append(3)
        filenames_new.append(filename)
    else :
        filenames.remove(filename)


df = pd.DataFrame({
    'filename': filenames_new,
    'category': categories
})
IMAGE_WIDTH=300
IMAGE_HEIGHT=300
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
#### final architecture if accuracy not increase  #####

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
np.random.seed(1000)

#Instantiation
AlexNet = Sequential()

#1st Convolutional Layer
AlexNet.add(Conv2D(filters=96, input_shape=(300,300,3), kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th Convolutional Layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))


#6th Convolutional Layer
AlexNet.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
# Add Dropout to prevent overfitting
AlexNet.add(Dropout(0.4))

#2nd Fully Connected Layer
AlexNet.add(Dense(2048))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#3rd Fully Connected Layer
AlexNet.add(Dense(1024))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#Output Layer
AlexNet.add(Dense(4))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

#Model Summary
AlexNet.summary()
AlexNet.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
#check dimensions of training set
#Onehot Encoding the labels.
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical
y_train = to_categorical(list(train_df['category']) , dtype = int)
y_val = to_categorical(list(validate_df['category']) , dtype = int)
x_train = train_df['filename']
x_val = validate_df['filename']
import cv2
#since we have filenames and we need numpy array for feeding in the network so converting image to numpy array
x_train_nparray=[]
for name in x_train:
    im = cv2.imread("../input/nnfl-lab-1/training/training/" + str(name))
    im = cv2.resize(im, IMAGE_SIZE,IMAGE_SIZE)
    x_train_nparray.append(np.asarray(im))

x_train_nparray = np.asarray(x_train_nparray)
print(x_train_nparray.shape)
#since we have filenames and we need numpy array for feeding in the network so converting image to numpy array on validation set
x_val_nparray=[]
for name in x_val:
    im = cv2.imread("../input/nnfl-lab-1/training/training/" + str(name))
    im = cv2.resize(im, IMAGE_SIZE,IMAGE_SIZE)
    x_val_nparray.append(np.asarray(im))

x_val_nparray = np.asarray(x_val_nparray)
print(x_val_nparray.shape)
#Image Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)

#Fitting the augmentation defined above to the data
train_generator.fit(x_train_nparray)
val_generator.fit(x_val_nparray)
#Learning Rate Annealer

from keras.callbacks import ReduceLROnPlateau
lrr= ReduceLROnPlateau(monitor='val_accuracy',   factor=.01,   patience=3,  min_lr=1e-5) 
#Defining the parameters
batch_size= 10
epochs=150
learn_rate=.001
import keras
from keras.callbacks import Callback
class new_callback(keras.callbacks.Callback):
    def epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy')> 0.95): # select the accuracy
            print("\n !!! 95% accuracy, no further training !!!")
            self.model.stop_training = True

earlystop = new_callback()
history = AlexNet.fit_generator(train_generator.flow(x_train_nparray, y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = x_train_nparray.shape[0]//batch_size, validation_data = val_generator.flow(x_val_nparray, y_val, batch_size=batch_size),validation_steps =x_val_nparray.shape[0]//batch_size, callbacks = [lrr , earlystop], verbose=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
AlexNet.save_weights("2017A7PS0968G.h5")
test_filenames = os.listdir("../input/nnfl-lab-1/testing/testing/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
x_test = test_df['filename']
import cv2
#since we have filenames and we need numpy array for feeding in the network so converting image to numpy array
x_test_nparray=[]
for name in x_test:
    im = cv2.imread("../input/nnfl-lab-1/testing/testing/" + str(name))
    im = cv2.resize(im, IMAGE_SIZE,IMAGE_SIZE)
    x_test_nparray.append(np.asarray(im))

x_test_nparray = np.asarray(x_test_nparray)
print(x_test_nparray.shape)
#Making prediction
y_pred=AlexNet.predict_classes(x_test_nparray)
test_df['category'] = y_pred
print(test_df.head())
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('_').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('2017A7PS0968G.csv', index=False)
from IPython.display import HTML 
import pandas as pd 
import numpy as np
import base64 

def create_download_link(df, title = "Download CSV file", filename = "2017A7PS0968G.csv"): 
    csv = df.to_csv(index=False) 
    b64 = base64.b64encode(csv.encode()) 
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission_df)