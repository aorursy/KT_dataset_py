import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Flatten,Conv2D, Dropout, Dense, GlobalAveragePooling2D
import tensorflow as tf
import keras
from numpy import expand_dims
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
path = '../input/lego-minifigures-classification/'
pic=cv2.imread('../input/lego-minifigures-classification/marvel/0007/002.jpg')
pic.shape
opened_dir = os.listdir(path)
print(opened_dir)
index = pd.read_csv(path + 'index.csv')
index.tail()
index.columns
index.drop('Unnamed: 0', axis=1, inplace=True)
metadata = pd.read_csv(path+'metadata.csv')
index['name']=None
index.head()
metadata.head()
for i, name in zip(metadata['class_id'],metadata['minifigure_name']):
    for sor, j in enumerate(index['class_id']):
        if i==j:
            index.iat[sor, 3]=name
        
index.tail(10)
valid = index.copy()
filt = index['train-valid']=='train'
index.where(filt, inplace=True)
filt1 = valid['train-valid']=='valid'
valid.where(filt1, inplace=True)
index.dropna(inplace=True, axis=0)
valid.dropna(inplace=True, axis=0)
img = load_img('../input/lego-minifigures-classification/marvel/0002/003.jpg')

data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(width_shift_range=[-100,100])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(height_shift_range=0.4)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(rotation_range=90)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()
batch= 15
size= 256
nb_classes=30
IN_SHAPE=(size,size,3)
Epoch= 100
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, width_shift_range=0.4, 
                                   height_shift_range=0.4,fill_mode="nearest", zoom_range=0.4, vertical_flip=True, horizontal_flip=True, brightness_range=[0.2,1.0])
valid_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_dataframe(dataframe=index, directory=path,
                                                   x_col='path', y_col='name', batch_size= batch,
                                                   shuffle=True, target_size=(size,size))
valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid, directory=path,
                                                   x_col='path', y_col='name', batch_size= batch,
                                                   shuffle=False, target_size=(size,size))
!pip install efficientnet
import efficientnet.tfkeras as efn
def get_model1():    
    base_model =  efn.EfficientNetB6(input_shape=IN_SHAPE, weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    #x = Dense(500,activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Flatten()(x)
    #x = Dense(150,activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)
model1 = get_model1()
    
model1.compile(optimizer='SGD', loss='categorical_crossentropy', metrics= 'accuracy')
history = model1.fit_generator(train_generator, epochs=Epoch, validation_data=valid_generator)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()
valid.head()
from sklearn.model_selection import train_test_split
test, _val = train_test_split(valid, test_size=0.5)
test.head()
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test, directory=path, x_col='path', y_col='name', batch_size= 1,
                                 shuffle=False, target_size=(size,size))
model1.evaluate_generator(generator=valid_generator)
test_generator.reset()
pred=model1.predict_generator(test_generator,verbose=1)
predicted_classes=np.argmax(pred,axis=1)
predicted_classes
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_classes]
labels
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.tail()
for pic, name in zip(results['Filename'], results['Predictions']): 
    img = load_img(path+pic)
    plt.imshow(img)
    print(name)   
    plt.show()

   
