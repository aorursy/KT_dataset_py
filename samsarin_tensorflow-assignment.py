!pip install tensorflow_datasets
import numpy as np

from tensorflow.keras.models import Sequential,Model,load_model

from tensorflow.keras.layers import Dense,Convolution2D,Flatten,MaxPooling2D,BatchNormalization,Dropout,GlobalMaxPool2D,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.datasets.fashion_mnist import load_data

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import LearningRateScheduler,TensorBoard,ReduceLROnPlateau

from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow as tf

from tensorflow.keras import layers



%matplotlib inline
(x_train,y_train),(x_test,y_test) = load_data()
print('This unique labels are: ',np.unique(y_train))
print('Number of training examples',x_train.shape[0])

print('Number of testing examples',x_test.shape[0])

print('Size of each example is: ',x_train[0].shape)
fig = plt.figure()

for i in range(1,101):

  ax1 = fig.add_subplot(10,10,i)

  plt.xticks([])

  plt.yticks([])

  ax1.imshow(x_train[i].reshape(28,28),cmap = 'gray')
(ds_train,d_info) = tfds.load('fashion_mnist', split='train[:80%]',with_info=True,

                               as_supervised=True)



(ds_valid,d_info) = tfds.load('fashion_mnist', split='train[-20%:]',with_info=True,

                               as_supervised=True)

(ds_test,d_info) = tfds.load('fashion_mnist',split = 'test',with_info=True,

                               as_supervised=True)
def normalize_img(image, label):

  image = tf.reshape(image,(784,))

  return tf.cast(image, tf.float32) / 255., label



ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.cache()

ds_train = ds_train.batch(128)

ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)





ds_valid = ds_valid.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_valid = ds_valid.batch(128)

ds_valid = ds_valid.cache()

ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)



ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_test = ds_test.batch(128)

ds_test = ds_test.cache()

ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)



training_epochs = 50
model1 = Sequential()

model1.add(Dense(32,kernel_initializer = 'normal', input_dim=784,activation='relu'))

model1.add(Dense(64,kernel_initializer = 'normal',activation = 'relu'))

model1.add(Dense(128,kernel_initializer = 'normal',activation = 'relu'))

model1.add(Dense(10,activation='softmax'))

# Compile model

adam1 = optimizers.Adam(lr = 0.0001)

model1.compile(loss='sparse_categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])



history1 = model1.fit(ds_train,validation_data = ds_valid, epochs = training_epochs)
testing_loss1,testing_acc1 = model1.evaluate(ds_test)



print('Testing loss of this model is ',testing_loss1)

print('Testing Accuracy of this model is ',testing_acc1)
values1 = history1.history



training_accuracy_1 = values1['accuracy']

training_loss_1 = values1['loss']

validation_accuracy_1 = values1['val_accuracy']

val_loss_1 = values1['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_1,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_1,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_1,label = 'Training Loss')

plt.plot(epochs,val_loss_1, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
(ds_train,d_info) = tfds.load('mnist', split='train[:80%]',with_info=True,

                               as_supervised=True)



(ds_valid,d_info) = tfds.load('mnist', split='train[-20%:]',with_info=True,

                               as_supervised=True)

(ds_test,d_info) = tfds.load('mnist',split = 'test',with_info=True,

                               as_supervised=True)



#ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.cache()

ds_train = ds_train.batch(128)

ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)





#ds_valid = ds_valid.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_valid = ds_valid.batch(128)

ds_valid = ds_valid.cache()

ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)



#ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_test = ds_test.batch(128)

ds_test = ds_test.cache()

ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model2 = Sequential()

model2.add(Convolution2D(32,(3,3),activation = 'relu',input_shape = (28,28,1)))

model2.add(MaxPooling2D(2,2))

model2.add(Convolution2D(64,(3,3),activation = 'relu'))

model2.add(MaxPooling2D(2,2))

model2.add(Flatten())

model2.add(Dense(128,activation = 'relu'))

model2.add(Dropout(0.2))

model2.add(Dense(10,activation = 'softmax'))

model2.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])



history2 = model2.fit(ds_train,validation_data = ds_valid, epochs = training_epochs)
testing_loss2,testing_acc2 = model2.evaluate(ds_test)



print('Testing loss of this model is ',testing_loss2)

print('Testing Accuracy of this model is ',testing_acc2)
values2 = history2.history



training_accuracy_2 = values2['accuracy']

training_loss_2 = values2['loss']

validation_accuracy_2 = values2['val_accuracy']

val_loss_2 = values2['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_2,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_2,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_2,label = 'Training Loss')

plt.plot(epochs,val_loss_2, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
(x_train,y_train),(x_test,y_test) = load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
train_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True,vertical_flip=True,shear_range=0.2,zoom_range=0.2)

test_generator = ImageDataGenerator(rescale = 1./255)
train_datagen = train_generator.flow(x_train,y_train,batch_size=32)

test_datagen = test_generator.flow(x_test,y_test,batch_size=32)
model3 = Sequential()

model3.add(Convolution2D(32,(3,3),activation = 'relu',input_shape = (28,28,1)))

model3.add(MaxPooling2D(2,2))

model3.add(Convolution2D(64,(3,3),activation = 'relu'))

model3.add(MaxPooling2D(2,2))

model3.add(Flatten())

model3.add(Dense(128,activation = 'relu'))

model3.add(Dropout(0.2))

model3.add(Dense(10,activation = 'softmax'))

model3.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])



history3 = model3.fit_generator(train_datagen,steps_per_epoch = 60000//32,validation_data = test_datagen, validation_steps = 10000//32,epochs = training_epochs)
x_test = x_test/255

testing_loss3,testing_acc3 = model3.evaluate(x_test,y_test)



print('Testing loss of this model is ',testing_loss3)

print('Testing Accuracy of this model is ',testing_acc3)
values3 = history3.history



training_accuracy_3 = values3['accuracy']

training_loss_3 = values3['loss']

validation_accuracy_3 = values3['val_accuracy']

val_loss_3 = values3['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_3,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_3,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_3,label = 'Training Loss')

plt.plot(epochs,val_loss_3, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
(x_train,y_train),(x_test,y_test) = load_data()

x_train = x_train.reshape(x_train.shape[0],784)

x_test = x_test.reshape(x_test.shape[0],784)

x_train=np.dstack([x_train] * 3)

x_test=np.dstack([x_test]*3)

x_train = x_train.reshape(-1, 28,28,3)

x_test = x_test.reshape (-1,28,28,3)

x_train.shape,x_test.shape
# Resize the images 48*48 as required by VGG16





x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])

x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])



x_train.shape, x_test.shape
x_train = x_train / 255.

x_test = x_test / 255.

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



x_train,x_valid,y_train,valid_label = train_test_split(x_train,

                                                           y_train,

                                                           test_size=0.2,

                                                           random_state=13

                                                           )



x_train = preprocess_input(x_train)

x_valid = preprocess_input(x_valid)

x_test  = preprocess_input (x_test)
conv_base = VGG16(weights='../input/keras-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False, 

                  input_shape=(48, 48, 3)

                 )

#conv_base.summary()
train_features = conv_base.predict(np.array(x_train), batch_size=32, verbose=1)

test_features = conv_base.predict(np.array(x_test), batch_size=32, verbose=1)

val_features = conv_base.predict(np.array(x_valid), batch_size=32, verbose=1)
np.savez("train_features", train_features, y_train)

np.savez("test_features", test_features, y_test)

np.savez("val_features", val_features, valid_label)
train_features_flat = np.reshape(train_features, (48000, 1*1*512))

test_features_flat = np.reshape(test_features, (10000, 1*1*512))

val_features_flat = np.reshape(val_features, (12000, 1*1*512))
model4 = Sequential()

model4.add(Dense(32, activation='relu', input_dim=(1*1*512)))

model4.add(Dense(64,activation = 'relu'))

model4.add(Dense(64,activation = 'relu'))

model4.add(Dropout(0.2))

model4.add(Dense(10, activation='softmax'))



model4.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history4 = model4.fit(train_features_flat,y_train,epochs=training_epochs,validation_data=(val_features_flat, valid_label))
testing_loss4,testing_acc4 = model4.evaluate(test_features_flat,y_test)



print('Testing loss of this model is ',testing_loss4)

print('Testing Accuracy of this model is ',testing_acc4)
values4 = history4.history



training_accuracy_4 = values4['accuracy']

training_loss_4 = values4['loss']

validation_accuracy_4 = values4['val_accuracy']

val_loss_4 = values4['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_4,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_4,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_4,label = 'Training Loss')

plt.plot(epochs,val_loss_4, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
(x_train,y_train),(x_test,y_test) = load_data()

x_train = x_train.reshape(x_train.shape[0],784)

x_test = x_test.reshape(x_test.shape[0],784)

x_train=np.dstack([x_train] * 3)

x_test=np.dstack([x_test]*3)

x_train = x_train.reshape(-1, 28,28,3)

x_test = x_test.reshape (-1,28,28,3)







x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])

x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])





x_train = x_train / 255

x_test = x_test / 255

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



conv_base = VGG16(weights='../input/keras-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False, 

                  input_shape=(48, 48, 3)

                 )

model5 = Sequential()

model5.add(conv_base)

model5.add(GlobalMaxPool2D())

model5.add(Flatten())

model5.add(Dense(128,activation = 'relu'))

model5.add(Dense(10,activation = 'softmax'))



model5.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])



history5 = model5.fit(x_train,y_train,validation_split = 0.2,batch_size = 32,epochs = training_epochs)
testing_loss5,testing_acc5 = model5.evaluate(x_test,y_test)



print('Testing loss of this model is ',testing_loss5)

print('Testing Accuracy of this model is ',testing_acc5)
values5 = history5.history



training_accuracy_5 = values5['accuracy']

training_loss_5 = values5['loss']

validation_accuracy_5 = values5['val_accuracy']

val_loss_5 = values5['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_5,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_5,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_5,label = 'Training Loss')

plt.plot(epochs,val_loss_5, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
!nvidia-smi
from tensorflow.keras.datasets.cifar10 import load_data

(x_train,y_train),(x_test,y_test) = load_data()



x_train = x_train/255

x_test = x_test/255



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



print('Shape of training data ',x_train.shape)

print('Shape of testing data ',x_test.shape)
model6 = Sequential()

model6.add(Convolution2D(32,(3,3),activation='relu',input_shape = (32,32,3)))

model6.add(MaxPooling2D(2,2))

model6.add(Convolution2D(32,(3,3),activation = 'relu'))

model6.add(MaxPooling2D(2,2))

model6.add(BatchNormalization())

model6.add(Convolution2D(64,(3,3),activation = 'relu'))

model6.add(Convolution2D(64,(3,3),activation='relu',padding='same'))

model6.add(MaxPooling2D(2,2))

model6.add(Flatten())

model6.add(Dense(128,activation = 'relu'))

model6.add(Dropout(0.2))

model6.add(Dense(10,activation = 'softmax'))



model6.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])



history6 = model6.fit(x_train,y_train,validation_split = 0.2,batch_size = 32,epochs = training_epochs)
testing_loss6,testing_acc6 = model6.evaluate(x_test,y_test)



print('Testing loss of this model is ',testing_loss6)

print('Testing Accuracy of this model is ',testing_acc6)
values6 = history6.history



training_accuracy_6 = values6['accuracy']

training_loss_6 = values6['loss']

validation_accuracy_6 = values6['val_accuracy']

val_loss_6 = values6['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_6,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_6,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_6,label = 'Training Loss')

plt.plot(epochs,val_loss_6, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
import tensorflow.keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)

from tensorflow.keras.initializers import glorot_uniform
def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    F1, F2, F3 = filters

    

    X_shortcut = X

    

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X
def convolutional_block(X, f, filters, stage, block, s = 2):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    F1, F2, F3 = filters

    

    X_shortcut = X





    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)



    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X
def ResNet50(input_shape = (32, 32, 3), classes = 14):

    X_input = Input(input_shape)



    

    X = ZeroPadding2D((3, 3))(X_input)

    

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3))(X)



    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')

    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')



    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')

    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')



    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')

    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')



    X = convolutional_block(X, f = 3, filters = [256,256, 1024], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [256,256, 1024], stage=5, block='b')

    X = identity_block(X, 3, [256,256, 1024], stage=5, block='c')



    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)

    



    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    

    

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
model7 = ResNet50(input_shape = (32, 32, 3), classes = 10)

model7.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
%%time

history7 = model7.fit(x_train,y_train,validation_split = 0.2,batch_size = 32,epochs = training_epochs)
testing_loss7,testing_acc7 = model7.evaluate(x_test,y_test)



print('Testing loss of this model is ',testing_loss7)

print('Testing Accuracy of this model is ',testing_acc7)
values7 = history7.history



training_accuracy_7 = values7['accuracy']

training_loss_7 = values7['loss']

validation_accuracy_7 = values7['val_accuracy']

val_loss_7 = values7['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_7,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_7,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_7,label = 'Training Loss')

plt.plot(epochs,val_loss_7, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model8 = ResNet50(input_shape = (32, 32, 3), classes = 10)

def scheduler4(epoch):

  if epoch < 20:

    return 0.001

  else:

    return 0.01 * tf.math.exp(0.5 * (10 - epoch))



callback = LearningRateScheduler(scheduler4)



model8.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
%%time

history8 = model8.fit(x_train,y_train,validation_split = 0.2,batch_size=32,epochs = training_epochs,callbacks=[callback])
testing_loss8,testing_acc8 = model8.evaluate(x_test,y_test)



print('Testing loss of this model is ',testing_loss8)

print('Testing Accuracy of this model is ',testing_acc8)
values8 = history8.history



training_accuracy_8 = values8['accuracy']

training_loss_8 = values8['loss']

validation_accuracy_8 = values8['val_accuracy']

val_loss_8 = values8['val_loss']

epochs = range(1,training_epochs+1)



plt.plot(epochs,training_accuracy_8,label = 'Training Accuracy')

plt.plot(epochs,validation_accuracy_8,label = 'Validation Accuracy')

plt.title('Accuracy vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



plt.plot(epochs,training_loss_8,label = 'Training Loss')

plt.plot(epochs,val_loss_8, label = 'Validation Loss')

plt.title('Loss vs Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()