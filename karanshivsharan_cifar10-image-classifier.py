import pandas as pd
import numpy as np # linear algebra
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
batch_1=unpickle('../input/cifar10/data_batch_1')
batch_2=unpickle('../input/cifar10/data_batch_2')
batch_3=unpickle('../input/cifar10/data_batch_3')
batch_4=unpickle('../input/cifar10/data_batch_4')
batch_5=unpickle('../input/cifar10/data_batch_5')
test_batch=unpickle('../input/cifar10/test_batch')
batch1_data=batch_1[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
batch2_data=batch_2[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
batch3_data=batch_3[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
batch4_data=batch_4[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
batch5_data=batch_5[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)
test_data=test_batch[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1)

batch1_labels=batch_1[b'labels']
batch2_labels=batch_2[b'labels']
batch3_labels=batch_3[b'labels']
batch4_labels=batch_4[b'labels']
batch5_labels=batch_5[b'labels']
test_labels=test_batch[b'labels']

train_images=np.concatenate((batch1_data,batch2_data,batch3_data,batch4_data,batch5_data),axis=0)
train_labels_data=np.concatenate((batch1_labels,batch2_labels,batch3_labels,batch4_labels,batch5_labels),axis=0)
train_images.shape
plt.imshow(train_images[5],cmap='binary')
train_labels_data[5]
sns.countplot(train_labels_data)
#labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
plt.imshow(train_images[1],cmap='binary')
train_labels_data[1]
def label(num):
  if num==0:
    return 'Airplane'
  elif num==1:
    return 'Automobile'
  elif num==2:
    return 'Bird'
  elif num==3:
    return 'Cat'
  elif num==4:
    return 'Deer'
  elif num==5:
    return 'Dog'
  elif num==6:
    return 'Frog'
  elif num==7:
    return 'Horse'
  elif num==8:
    return 'Ship'
  elif num==9:
    return 'Truck'
plt.subplot(1,2,1)
plt.imshow(train_images[50],cmap='binary')
print(f'The image is of {label(train_labels_data[50])}')
plt.subplot(1,2,2)
plt.imshow(test_data[50],cmap='binary')
print(f'The image is of {label(test_labels[50])}')
train_labels_data
from keras.utils import to_categorical
train_label=to_categorical(train_labels_data)
Test_labels=to_categorical(test_labels)
datagen=ImageDataGenerator( featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
train_data=datagen.flow(train_images, train_label, batch_size=32)
validation_data=datagen.flow(test_data, Test_labels, batch_size=32)
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(input_shape=[32,32,3],filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(strides=2,pool_size=(2,2)))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(input_shape=[32,32,3],filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(strides=2,pool_size=(2,2)))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(input_shape=[32,32,3],filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(strides=2,pool_size=(2,2)))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(784,activation='relu'))
cnn.add(tf.keras.layers.Dense(784,activation='relu'))
cnn.add(tf.keras.layers.Dense(10,activation='softmax'))
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(train_data,epochs=20)
prediction=cnn.predict_classes(test_data)
target_label=np.argmax(Test_labels,1)
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(prediction,target_label),annot=True)
from sklearn.metrics import classification_report
print(classification_report(prediction,target_label))
img=image.load_img('../input/cifar10/plane.jpg',target_size=(32,32))
raw_image=image.img_to_array(img)
test_image=np.expand_dims(raw_image,axis=0)
print(f'The Prediction of image is {label(cnn.predict_classes(test_image))}')
ti=plt.imread('../input/cifar10/plane.jpg')
plt.imshow(ti)

