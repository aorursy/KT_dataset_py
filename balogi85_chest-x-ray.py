import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dropout, Dense, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.applications import Xception, ResNet50, MobileNet, DenseNet201, VGG16
from keras import Model
path = '../input/chest-xray-pneumonia/chest_xray/'
train_dir = path + 'train/'
val_dir = path + 'test/'
test_dir = path + 'val/'
print(os.listdir(train_dir))
im = Image.open('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
print(im.size)
print(im.mode)
norm = Image.open('../input/chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg')
pneum = Image.open('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg')
pneum_bact = Image.open('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1006_bacteria_2937.jpeg')
pneum_virus = Image.open('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_virus_1681.jpeg')
f = plt.figure(figsize= (30,10))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(pneum)
a2.set_title('Pneumonia')

f = plt.figure(figsize= (30,10))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(pneum_bact)
a1.set_title('Bacterical')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(pneum_virus)
a2.set_title('Virus')

train_data_gen = ImageDataGenerator(rescale=1.0/255, zoom_range=0.1,rotation_range=0.1, vertical_flip=True)
val_data_gen = ImageDataGenerator(rescale = 1.0/255)
test_data_gen = ImageDataGenerator(rescale = 1.0/255)
size = 224
epoch = 20
color = 'rgb'
application = [Xception, ResNet50, MobileNet, DenseNet201, VGG16]
train_generator = train_data_gen.flow_from_directory(train_dir,target_size=(size, size),color_mode=color, batch_size=32, class_mode='binary')
valid_generator = val_data_gen.flow_from_directory(val_dir, target_size=(size,size),color_mode=color, batch_size=32, class_mode='binary')
test_generator = test_data_gen.flow_from_directory(test_dir, target_size=(size,size),color_mode=color, batch_size=32, class_mode='binary')
def get_model(app):
    base_model =  app(input_shape=(size, size, 3), weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    x = Dense(400,activation='relu')(x)
    x= Dropout(0.2)(x)
    x = Dense(300,activation='relu')(x)
    x= Dropout(0.2)(x)
    x = Dense(150,activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=predictions)
model= get_model(application[4])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics= 'accuracy')
model.summary()

history = model.fit_generator(train_generator, epochs=epoch, validation_data=valid_generator)
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
model.save_weights("vgg16.h5")
