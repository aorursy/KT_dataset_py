import os
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from collections import Counter
import random
train_images=pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/train/train.csv')
train_images.head()
#Is the problem balanced or imbalanced classification
plt.figure(figsize=(6,5))
count=Counter(train_images['emergency_or_not'])
print(count)
ax=sb.barplot(list(count.keys()),list(count.values()))
for i,j in enumerate(count.values()):
    ax.text(i,j+5,'{}'.format(j)) #Pretty much balanced i can use accuracy as a metric
test_images=pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/test.csv')
test_images.head()
print('Number of train points = ',train_images.shape[0])
print('Percentage of train points =',train_images.shape[0]/(train_images.shape[0]+test_images.shape[0]))
print('No of test points =',test_images.shape[0])
print('Percentage of test points =',test_images.shape[0]/(train_images.shape[0]+test_images.shape[0]))
file_names=os.listdir('../input/emergency-vs-nonemergency-vehicle-classification/train/images')
Y=train_images['emergency_or_not'].values
labels=[]
for i in Y:
    if(i==0):
        labels.append('Non-emergency')
    else:
        labels.append('Emergency')
import skimage
from skimage.io import imread
#Plotting some images from train data
plt.figure(figsize=(10,10))
for i,j in enumerate(train_images['image_names'].values):
    if(i>=25):
        break
    plt.subplot(5,5,1+i)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    a=imread('../input/emergency-vs-nonemergency-vehicle-classification/train/images/'+j)
    plt.imshow(a)
    plt.xlabel(labels[i])
#Loading images and Converting them to arrays
def img2array(a1):
    array=[]
    for i in a1.values:
        img=tf.keras.preprocessing.image.load_img('../input/emergency-vs-nonemergency-vehicle-classification/train/images/'+i)
        img=tf.keras.preprocessing.image.img_to_array(img)
        array.append(img)
    return(array)
X=img2array(train_images['image_names'])
X_test=img2array(test_images['image_names'])
X_test=np.array(X_test)
print(X_test.shape)
X_test=X_test/255
X=np.array(X)
print(X.shape)
X=X/255 
#Deep learning model building

#Approach1:Without image augmentation,Without l2 regularization or dropout
from tensorflow.keras import layers,models
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(2,2),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
#Number of parameters to be learnt at each layer=number of filters*filtersize(i.e filter_height*filter_width*number of channels)+no of filters
#Since the problem is about binary classification i.e predicting emergency/not,hence we use binarycrossentropy as loss function
#Problem also has balanced target variable so im using accuracy as metric
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=False),metrics=['accuracy'])
callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
history=model.fit(X,Y,epochs=10,validation_split=0.15,callbacks=callback)
plt.plot(range(1,6),history.history['accuracy'],label='train_accuracy')
plt.plot(range(1,6),history.history['val_accuracy'],label='validation_accuracy')
plt.legend()
plt.show() #There is a huge overfitting happening with this approach
#Approach2: Using l2 regularization 
from tensorflow.keras import regularizers

model1=models.Sequential()
model1.add(layers.Conv2D(32,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(224,224,3)))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(64,(2,2),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model1.add(layers.MaxPooling2D(2,2))
model1.add(layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(1,activation='sigmoid'))
model1.summary()
model1.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True),metrics=['accuracy'])
model1.fit(X,Y,epochs=10,validation_split=0.15,callbacks=callback)
prediction_with_l2=model1.predict(X_test,batch_size=32)
test_images=test_images.assign(predictions_with_l2=prediction_with_l2)
test_images.head().to_csv()
#Approach3:Using l2regularization+dropout
model2=models.Sequential()
model2.add(layers.Conv2D(32,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(224,224,3)))
model2.add(layers.Dropout(0.5))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Conv2D(64,(2,2),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Conv2D(64,(3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Flatten())
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(1,activation='sigmoid'))
model2.summary()
model2.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=False),metrics=['accuracy'])
hsitory2=model2.fit(X,Y,epochs=10,validation_split=0.15,callbacks=[callback])
# Approach4:Leaky Relu+Weight initilaization using glorot_normal
model3=models.Sequential()
model3.add(layers.Conv2D(32,(3,3),kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(224,224,3)))
model3.add(layers.Dropout(0.5))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Conv2D(64,(2,2),kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model3.add(layers.Dropout(0.5))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Conv2D(64,(3,3),kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model3.add(layers.Dropout(0.5))
model3.add(layers.Flatten())
model3.add(layers.Dense(64))
model3.add(layers.Dense(1,activation='sigmoid'))
model3.summary()
model3.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=False),metrics=['accuracy'])
history3=model3.fit(X,Y,epochs=10,validation_split=0.15,callbacks=callback)
plt.plot(range(1,11),history3.history['accuracy'],label='train')
plt.plot(range(1,11),history3.history['val_accuracy'],label='val')
plt.legend()
plt.show()
#Trying Image Augmentation for increasing accuracy
imagegen=tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=1, width_shift_range=0.0,
    height_shift_range=0.3, brightness_range=(0.3,2), shear_range=0.4, zoom_range=(0.8,2),
    channel_shift_range=0.0, horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last', validation_split=0.15, dtype=None
)
#Checking Imagedata Augmentation
imagegen1=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)
z=np.expand_dims(X[12],0)
it=imagegen.flow(z,batch_size=1)
it
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,1+i)
    plt.xticks([])
    plt.yticks([])
    batch=it.next()
    bat=batch[0].astype('uint8')
    plt.imshow(bat)
#Using Transfer learning to improve accuracy
x_tensor=tf.data.Dataset.from_tensor_slices(X)
x_tensor=x_tensor.map(lambda x: tf.image.resize(x,[160,160]))
n=list(x_tensor.as_numpy_iterator())
n=np.array(n)
print(n.shape) # Image resized/reshaped successfully from (224,224) to (160,160)
base_model = tf.keras.applications.MobileNetV2(input_shape=(160,160,3),
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()
base_model.trainable=False
model5=models.Sequential()
model5.add(base_model)
model5.add(layers.Flatten())
model5.add(layers.Dense(1))
model5.summary() #If we look out of 2,289,985 parameters only 32,001 are trainable which means above layers of pretrained model are freezed 
model5.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
history5=model5.fit(n,Y,batch_size=32,epochs=10,validation_split=0.15,callbacks=[callback])
print(len(base_model.layers))
history5.epoch[-1]
#Fine tuning approach
base_model.trainable=True
for layer in base_model.layers[:100]:
    layer.trainable=False            #Here freezing all the top 100 features,the rest of the layers are where complex features are learned
model5.summary()#As we see by reactivating last 55 layers in pretrained model the params increased to 1,894,593
total_epoch=20
model5.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
history6=model5.fit(n,Y,batch_size=32,epochs=total_epoch,initial_epoch=history5.epoch[-1],validation_split=0.15)