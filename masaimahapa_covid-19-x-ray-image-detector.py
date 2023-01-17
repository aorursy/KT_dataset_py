# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print(os.listdir('../input/covid19/'))
corona_path='../input/covid19/AI4COVID-19 Hackathon Dataset/Train/Positives/covid-19-pneumonia-19.jpg'
negative_path='../input/covid19/AI4COVID-19 Hackathon Dataset/Train/Negatives/acute-respiratory-distress-syndrome-ards.jpg'
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img = mpimg.imread(corona_path)

imgplot = plt.imshow(img)
img = mpimg.imread(negative_path)
imgplot = plt.imshow(img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (256, 256)
batch_size=16
datagen = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True,
                             
                              horizontal_flip = True, 
                              vertical_flip = True, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=180, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
train_generator= datagen.flow_from_directory(
    '../input/covid19/AI4COVID-19 Hackathon Dataset/Train',
    target_size= IMG_SIZE, 
    color_mode='rgb',
    batch_size=batch_size, 
    class_mode='binary'
)

validation_generator= next(datagen.flow_from_directory(
    '../input/covid19/AI4COVID-19 Hackathon Dataset/Validation/',
    target_size= IMG_SIZE,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary'
    
))#makes one big batch

x_test= next(datagen.flow_from_directory(
    '../input/covid-test/covid_test',
    target_size=IMG_SIZE,
    color_mode= 'rgb',
    batch_size=batch_size,
    class_mode=None
))#makes one big batch
base_model= ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))

base_model.summary()
from tensorflow.keras.optimizers import SGD

base_model.trainable = False

model = Sequential()

model.add(base_model)
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy', 'mae'])
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

weight_path= '{}_wights.best.hdf5'.format('covid_cnn')

checkpoint= ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True,
                           mode='min', save_weights_only=True)

early= EarlyStopping(monitor='val_loss', mode='min', patience=30)

callbacks_list= [checkpoint, early]
history= model.fit_generator(train_generator,
                    
                   validation_data=validation_generator, epochs=1, callbacks= callbacks_list,
                            shuffle=True)

model.save('corona_cnn')
history= model.fit_generator(train_generator,
                   validation_data=validation_generator,
                    epochs=100,
                    callbacks= callbacks_list,
                            shuffle=True)
def plot_learning_curves(history):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1,3,2)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('binary accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1,3,3)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.tight_layout()
    
plot_learning_curves(history)
#predictions
y_preds= model.predict(validation_generator, verbose=True)
y_preds=np.round(y_preds)

#true labels
y_val=validation_generator[1]
from sklearn.metrics import classification_report, confusion_matrix
print('confusion matrix')
print(confusion_matrix(y_preds, y_val))
print('------------------------------------------')
print('classification_report')
print(classification_report(y_preds, y_val))
model.load_weights(weight_path)
scores = model.evaluate(validation_generator[0], validation_generator[1])
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("val_loss:", scores[0])
print("val_mean_absolute_error:", scores[2])
df= pd.DataFrame(y_preds, columns=['predictions'])
df.to_csv('corona_predictions.csv')