# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import ResNet50, Xception, InceptionV3 , VGG16
from keras.layers import *
from keras import regularizers, optimizers
import keras.backend as K
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions

train_img_dir_path = "/kaggle/input/gulf-vehicle-plate/vlpc_train/"
train_dataset_file =  "/kaggle/input/gulf-vehicle-plate/vlpc_train.csv"
test_img_dir_path = "/kaggle/input/gulf-vehicle-plate/vlpc_test/"
test_dataset_file =  "/kaggle/input/gulf-vehicle-plate/vlpc_test.csv"
image_width = 120
image_height = 90
image_size = ( image_height , image_width )
image_channels = 3
batch_size = 10
def convert_country_code(plate_df):
    plate_df.vehicle_country_code = pd.Categorical(plate_df.vehicle_country_code)
    plate_df['vehicle_country_id'] = plate_df.vehicle_country_code.cat.codes
    plate_df['vehicle_country_id'] = plate_df['vehicle_country_id'].astype(str)
    return plate_df;
#reading the csv file , shuffle , encode category (not binary so we can extend it for other gulf plate types)
plate_img_df = pd.read_csv(train_dataset_file)   
plate_img_df = shuffle(plate_img_df)
plate_img_df.head()  
#convert country category column
plate_img_df = convert_country_code(plate_img_df)
plate_img_df.dtypes
# split dataset to train , validation 
train_df, validation_df = train_test_split(plate_img_df , test_size=0.20, random_state=42)
print(" Training KSA Plate Count  ",train_df[train_df['vehicle_country_code'] == 'KSA']['vehicle_country_code'].count())
print(" Training BAH Plate Count  ",train_df[train_df['vehicle_country_code'] == 'BAH']['vehicle_country_code'].count())
print(" Validation Set KSA Plate Count  ",validation_df[validation_df['vehicle_country_code'] == 'KSA']['vehicle_country_code'].count())
print(" Validation Set BAH Plate Count  ",validation_df[validation_df['vehicle_country_code'] == 'BAH']['vehicle_country_code'].count())
# data augmentation via ImageDataGenerator class
train_datagen = image.ImageDataGenerator(    
    rotation_range = 15,             
    rescale = 1./255,              
    shear_range = 0.1,             
    zoom_range = 0.2,                
    horizontal_flip = True,          
    width_shift_range = 0.1,         
    height_shift_range = 0.1,
    fill_mode = 'nearest'
)

#Note that validation data shouldn’t be augmented
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    directory = train_img_dir_path ,
    dataframe = train_df,
    x_col = 'vehicle_plate_image_id',
    y_col = 'vehicle_country_id',
    target_size=(image_height, image_width),
    class_mode = 'categorical',  
    batch_size = batch_size,
)

validation_generator = validation_datagen.flow_from_dataframe(
    directory = train_img_dir_path ,
    dataframe = validation_df,
    x_col = 'vehicle_plate_image_id',
    y_col = 'vehicle_country_id',
    target_size=(image_height, image_width),
    class_mode = 'categorical',  
    batch_size = batch_size,
)
#sanity check
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
#Build Model gulf_vehicle_country_model from scratch
gulf_vehicle_country_model = Sequential()
gulf_vehicle_country_model.add(Conv2D(32, (3, 3), activation='relu',input_shape=( image_height , image_width , image_channels )))
gulf_vehicle_country_model.add(MaxPooling2D((2, 2)))
gulf_vehicle_country_model.add(Conv2D(32, (3, 3), activation='relu'))
gulf_vehicle_country_model.add(MaxPooling2D((2, 2)))
gulf_vehicle_country_model.add(BatchNormalization())
gulf_vehicle_country_model.add(Flatten())
#gulf_vehicle_country_model.add(Dropout(0.4))
gulf_vehicle_country_model.add(Dense(32, activation='relu'))
gulf_vehicle_country_model.add(Dense(2, activation='softmax'))
gulf_vehicle_country_model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
gulf_vehicle_country_model.summary()
#train cnn_model scratch model
custom_cnn_history = gulf_vehicle_country_model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=20,
validation_data=validation_generator,
validation_steps=50)
acc = custom_cnn_history.history['accuracy']
val_acc = custom_cnn_history.history['val_accuracy']
loss = custom_cnn_history.history['loss']
val_loss = custom_cnn_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
gulf_vehicle_country_model.save('gulf_vehicle_country_model_v1.h5')
#Use ResNet50 Model
conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=( image_height , image_width , image_channels))
conv_base.trainable = False
resnet50_model = Sequential()
resnet50_model.add(conv_base)
#resnet50_model.add(Dropout(0.6))
resnet50_model.add(BatchNormalization())
resnet50_model.add(Flatten())
resnet50_model.add(Dense(16 , activation='relu'))
resnet50_model.add(Dense(2, activation='softmax'))
resnet50_model.compile(loss='categorical_crossentropy', optimizer= optimizers.SGD(lr=0.0001) , metrics=['accuracy'])
resnet50_model.summary()
#train resnet50_model  model
resnet50_history = resnet50_model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=20,
validation_data=validation_generator,
validation_steps=50)
acc = resnet50_history.history['accuracy']
val_acc = resnet50_history.history['val_accuracy']
loss = resnet50_history.history['loss']
val_loss = resnet50_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Use InceptionV3 Model
conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=( image_height , image_width , image_channels))
conv_base.trainable = False
inception_v3_model = Sequential()
inception_v3_model.add(conv_base)
inception_v3_model.add(Dropout(0.6))
inception_v3_model.add(BatchNormalization())
inception_v3_model.add(Flatten())
inception_v3_model.add(Dense(32 , activation='relu'))
inception_v3_model.add(Dense(2, activation='softmax'))
inception_v3_model.compile(loss='categorical_crossentropy', optimizer= optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
inception_v3_model.summary()
#train inception_v3_model  
inception_v3_history = inception_v3_model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=20,
validation_data=validation_generator,
validation_steps=50)
inception_v3_model.save('inception_v3_model.h5')
acc = inception_v3_history.history['accuracy']
val_acc = inception_v3_history.history['val_accuracy']
loss = inception_v3_history.history['loss']
val_loss = inception_v3_history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Prepare Test dataset 
test_df = pd.read_csv(test_dataset_file)
test_df = shuffle(test_df)
test_df = convert_country_code(test_df)
x_test = []
y_test = []
for index, img_row in test_df.iterrows():
    img_file = test_img_dir_path + img_row['vehicle_plate_image_id']
    plate_img_file = image.load_img(img_file, target_size=image_size)
    img_tensor = image.img_to_array(plate_img_file)
    img_tensor = img_tensor/255
    #img_tensor = preprocess_input(img_tensor)
    x_test.append(img_tensor)
    y_test.append(img_row['vehicle_country_id'])
x_test = np.array(x_test, dtype="float") 
y_test = np.array(y_test)
#y_test = np.where(y_test == 'KSA', 1 , 0)
y_test_category = to_categorical(y_test, num_classes=2)
test_df.head()
#Evaluate model for test dataset
loss,accuracy =  gulf_vehicle_country_model.evaluate(x = x_test, y = y_test_category)
print("  Accuracy of  gulf_vehicle_country_model  ",accuracy*100 , "%")

#loss,accuracy =  inception_v3_model.evaluate(x = x_test, y = y_test_category)
#print("  Accuracy of  inception_v3_model ",accuracy*100 , "%")
# Test Sample 
img_path = test_img_dir_path + "KSA_299.jpg"
img = image.load_img(img_path, target_size=image_size)
img_tensor = image.img_to_array(img)
img_tensor = img_tensor/255
img_tensor = np.expand_dims(img_tensor, axis=0)
print(img_tensor.shape)
# Display Test Sample
plt.imshow(img_tensor[0])
plt.show()
#Load model & Predict
#cnn_model_test = load_model('custom_cnn_model_v1.h5')
preds = gulf_vehicle_country_model.predict_classes(img_tensor)
print('Predicted Class  ', preds)
#Load resnet50_model & Predict
img_tensor = preprocess_input(img_tensor)
resnet50_model_test = load_model('resnet50_model_v1.h5')
preds = resnet50_model_test.predict_classes(img_tensor)
print('Predicted:', preds)
#Load inception_v3_model & Predict
img_tensor = preprocess_input(img_tensor)
#inception_v3_model_test = load_model('inception_v3_model.h5')
preds = inception_v3_model.predict_classes(img_tensor)
print('Predicted:', preds)