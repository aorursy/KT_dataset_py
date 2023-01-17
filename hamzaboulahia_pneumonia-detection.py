# Basic libraries

import sys

import os

import gc

import time              

import pickle                

import numpy as np

import pandas as pd

from time import time



# Data visualization & printing libraries

from matplotlib import pyplot as plt

import seaborn as sb

from PIL import Image

from IPython.display import HTML, display

import tabulate



# Deep learning libraries

import tensorflow as tf

from keras import Model

from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from keras.layers import Flatten, Activation, Input, AveragePooling2D, Lambda, add

from keras.layers.merge import concatenate

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from keras.utils import plot_model



from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# Utility functions

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.utils import shuffle



# Setting random seeds

np.random.seed(7)

tf.random.set_seed(7)



# Magic functions

%matplotlib inline
# Checking that tesorflow is using the GPU



from tensorflow.python.client import device_lib 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')),'\n')

print(device_lib.list_local_devices(),'\n')

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# Getting image paths and creating pandas dataframe for each dataset (train/val/test)



train_p = [('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'))]

train_n = [('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/train/NORMAL'))]

val_p = [('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA'))]

val_n = [('../input/chest-xray-pneumonia/chest_xray/val/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/val/NORMAL'))]

test_p = [('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA'))]

test_n = [('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/NORMAL'))]



train_data = pd.DataFrame(train_p+train_n, columns=['image_path', 'label'],index=None)

val_data = pd.DataFrame(val_p+val_n, columns=['image_path', 'label'],index=None)

test_data = pd.DataFrame(test_p+test_n, columns=['image_path', 'label'],index=None)
print("The head of the training dataframe:")

train_data.head()
# Count plot of the class representations.



plt.figure(figsize=(15,4))

plt.subplot(1,3,1)

sb.countplot(data= train_data, x='label')

plt.title('Number of cases in training data', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(train_data.label.unique())), ['Normal', 'Pneumonia'])



plt.subplot(1,3,2)

sb.countplot(data= val_data, x='label')

plt.title('Number of cases in validation data', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(val_data.label.unique())), ['Normal', 'Pneumonia'])



plt.subplot(1,3,3)

sb.countplot(data= test_data, x='label')

plt.title('Number of cases in test data', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(test_data.label.unique())), ['Normal', 'Pneumonia']);
# Visualizing a sample of 4 training images.



random_imgs = np.random.randint(1,train_data.shape[0],4)

plt.figure(figsize=(15,4))

plt.suptitle('Sample of 4 training images with labels', y=0.95,fontsize=15)

for i, img in enumerate(train_data.iloc[random_imgs,0]):

    image_temp = Image.open(img)                                           # Conversion to Black & White

    plt.subplot(1,4,i+1)

    plt.imshow(image_temp, cmap='gray')

    if train_data.iloc[random_imgs[i],1] == 0:

        plt.xlabel('Normal')

    else:

        plt.xlabel('Pneumonia')
# Creating a dataframe for image properties. 



l=[]

path_list=train_p+train_n+val_p+val_n+test_p+test_n

for i in path_list:

    with Image.open(i[0]) as image_temp:

        l.append(list(np.asarray(image_temp).shape))

imgs_size_df = pd.DataFrame(l, columns=['height', 'width','channels'],index=None)

imgs_size_df.channels.fillna(1, inplace=True)
# Example of B&W image stored in RGB format



img = Image.open(path_list[imgs_size_df.query('channels==3').index[0]][0])

plt.imshow(img)

plt.title('Example of a Grayscale image stored as RGB image')

print('Image size:', np.asarray(img).shape)
plt.figure(figsize=(8,5))

sb.scatterplot(data=imgs_size_df, x='height', y='width', hue='channels', palette="deep")

plt.title('Distribution of the image sizes in the training dataset');
imgs_size_df.fillna(value=int(1), inplace=True)

imgs_size_df.query('channels==3').head()

print('There are', imgs_size_df.query('channels==3').shape[0], 'B&W images stored in RGB format.')
# Resizing all the images: New size 256x256



direct_lists=[train_p,train_n,val_p,val_n,test_p,test_n]

save_paths=['/kaggle/working/modified/train/PNEUMONIA/',

            '/kaggle/working/modified/train/NORMAL/',

            '/kaggle/working/modified/val/PNEUMONIA/',

            '/kaggle/working/modified/val/NORMAL/',

            '/kaggle/working/modified/test/PNEUMONIA/',

            '/kaggle/working/modified/test/NORMAL/']



for path in save_paths:

    os.makedirs(path)
save_prefix=['train_pneumonial','train_normal','val_pneumonial','val_normal','test_pneumonial','test_normal']

for direct,save_path,prefix in zip(direct_lists,save_paths, save_prefix):

    i=1

    for path in direct:

        with Image.open(path[0]) as img_temp:

            resized_gray = img_temp.resize((256,256)).convert('L')

            resized_gray.save(save_path+prefix+'_'+str(i)+'.jpg')

        i+=1
# Modified image paths



train_p = [('/kaggle/working/modified/train/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('/kaggle/working/modified/train/PNEUMONIA/'))]

train_n = [('/kaggle/working/modified/train/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('/kaggle/working/modified/train/NORMAL/'))]

val_p = [('/kaggle/working/modified/val/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('/kaggle/working/modified/val/PNEUMONIA/'))]

val_n = [('/kaggle/working/modified/val/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('/kaggle/working/modified/val/NORMAL/'))]

test_p = [('/kaggle/working/modified/test/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('/kaggle/working/modified/test/PNEUMONIA/'))]

test_n = [('/kaggle/working/modified/test/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('/kaggle/working/modified/test/NORMAL/'))]



train_data = pd.DataFrame(train_p+train_n, columns=['image', 'label'],index=None)

val_data = pd.DataFrame(val_p+val_n, columns=['image', 'label'],index=None)

test_data = pd.DataFrame(test_p+test_n, columns=['image', 'label'],index=None)
l=[]

paths = train_p+train_n+val_p+val_n+test_p+test_n

for i in paths:

    with Image.open(i[0]) as image_temp:

        l.append(list(np.asarray(image_temp).shape))

imgs_size_df = pd.DataFrame(l, columns=['height', 'width'],index=None)
print('All the images are now of size',imgs_size_df.height.value_counts().index[0],

      'x',imgs_size_df.width.value_counts().index[0],'x 1')
# Creating an image generator



ImageGen = ImageDataGenerator(

    rotation_range=10,

    brightness_range=[0.9,1.1],

    zoom_range = 0.15,

    width_shift_range=0.1, 

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False,

    fill_mode="nearest"

)
# Function that generates and save images by batch and up to a certain threshold.



def GenerateImages(data, save_dir, b_size, max_size):

    img_array_list=[]

    for path in data:

        with Image.open(path[0]) as image_temp:

            img_array = img_to_array(image_temp)

            img_array = img_array.reshape((1,)+img_array.shape)

            img_array_list.append(img_array)

    img_array_list = np.concatenate(img_array_list, axis=0)

    i=0

    for batch in ImageGen.flow(img_array_list, batch_size=b_size,

                            save_to_dir='/kaggle/working/modified/'+save_dir,

                            save_prefix='Aug', save_format='JPEG'):

        i+=1

        if i>(max_size-len(data))//b_size:

            break
os.makedirs('/kaggle/working/modified/train/AUGMENTED/PNEUMONIA')

os.makedirs('/kaggle/working/modified/train/AUGMENTED/NORMAL')

os.makedirs('/kaggle/working/modified/val/AUGMENTED/PNEUMONIA')

os.makedirs('/kaggle/working/modified/val/AUGMENTED/NORMAL')
# Generating train and validation data



GenerateImages(train_p, 'train/AUGMENTED/PNEUMONIA', 16, 8000)

GenerateImages(train_n, 'train/AUGMENTED/NORMAL', 16, 8000)

GenerateImages(val_p, 'val/AUGMENTED/PNEUMONIA', 8, 400)

GenerateImages(val_n, 'val/AUGMENTED/NORMAL', 8, 400)
# Getting the augmented data paths and adding them to the old ones



train_p_aug = [('/kaggle/working/modified/train/AUGMENTED/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('/kaggle/working/modified/train/AUGMENTED/PNEUMONIA'))]

train_p = train_p + train_p_aug

train_n_aug = [('/kaggle/working/modified/train/AUGMENTED/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('/kaggle/working/modified/train/AUGMENTED/NORMAL'))]

train_n = train_n + train_n_aug



val_p_aug = [('/kaggle/working/modified/val/AUGMENTED/PNEUMONIA/' + filename,1) for count, filename in enumerate(os.listdir('/kaggle/working/modified/val/AUGMENTED/PNEUMONIA'))]

val_p = val_p + val_p_aug

val_n_aug = [('/kaggle/working/modified/val/AUGMENTED/NORMAL/' + filename,0) for count, filename in enumerate(os.listdir('/kaggle/working/modified/val/AUGMENTED/NORMAL'))]

val_n = val_n + val_n_aug
train_data = pd.DataFrame(train_p+train_n, columns=['image', 'label'],index=None)

val_data = pd.DataFrame(val_p+val_n, columns=['image', 'label'],index=None)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

sb.countplot(data= train_data, x='label')

plt.title('Number of cases in the training set', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(train_data.label.unique())), ['Normal', 'Pneumonia'])



plt.subplot(1,2,2)

sb.countplot(data= val_data, x='label')

plt.title('Number of cases in the validation set', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(train_data.label.unique())), ['Normal', 'Pneumonia']);
Train_data=[]

Train_label=[]

for path in train_n+train_p:

    with Image.open(path[0]) as image_temp:

        img_array = img_to_array(image_temp)

        img_array = (img_array-img_array.mean())/img_array.std()

        img_array = img_array.reshape((1,)+img_array.shape)

        Train_data.append(img_array)

    Train_label.append(path[1])

Train_data = np.concatenate(Train_data, axis=0)

Train_label = np.array(Train_label)



Val_data=[]

Val_label=[]

for path in val_n+val_p:

    with Image.open(path[0]) as image_temp:

        img_array = img_to_array(image_temp)

        img_array = (img_array-img_array.mean())/img_array.std()

        img_array = img_array.reshape((1,)+img_array.shape)

        Val_data.append(img_array)

    Val_label.append(path[1])

Val_data = np.concatenate(Val_data, axis=0)

Val_label = np.array(Val_label)



Test_data=[]

Test_label=[]

for path in test_n+test_p:

    with Image.open(path[0]) as image_temp:

        img_array = img_to_array(image_temp)

        img_array = (img_array-img_array.mean())/img_array.std()

        img_array = img_array.reshape((1,)+img_array.shape)

        Test_data.append(img_array)

    Test_label.append(path[1])

Test_data = np.concatenate(Test_data, axis=0)

Test_label = np.array(Test_label)
Train_data, Train_label = shuffle(Train_data, Train_label, random_state=7)

Val_data, Val_label = shuffle(Val_data, Val_label, random_state=7)

Test_data, Test_label = shuffle(Test_data, Test_label, random_state=7)
VGG_model = Sequential(name='VGG_model')

VGG_model.add(Conv2D(16,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal',

                       input_shape=(256,256,1) ))

VGG_model.add(MaxPooling2D((2,2), padding='same'))



VGG_model.add(Conv2D(32,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(Conv2D(32,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(BatchNormalization())

VGG_model.add(MaxPooling2D((2,2), strides=2 ,padding='same'))



VGG_model.add(Conv2D(64,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(Conv2D(64,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(BatchNormalization())

VGG_model.add(MaxPooling2D((2,2), strides=2 ,padding='same'))



VGG_model.add(Conv2D(128,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(Conv2D(128,(3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' ))

VGG_model.add(BatchNormalization())

VGG_model.add(MaxPooling2D((2,2), strides=2 ,padding='same'))



VGG_model.add(Flatten())



VGG_model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))

VGG_model.add(Dropout(0.15))

VGG_model.add(Dense(1, activation="sigmoid"))



VGG_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])



VGG_model.summary()
plot_model(VGG_model, show_shapes=True, to_file='VGG_model.png')
callback = ReduceLROnPlateau(monitor='val_loss', patience = 3, cooldown=0, verbose=1, factor=0.6, min_lr=0.000001)

start_vgg = time()

VGG_results = VGG_model.fit(

    x=Train_data,

    y=Train_label,

    batch_size=16,

    validation_data=(Val_data,Val_label),

    class_weight={0:12, 1:0.5},

    epochs=20, callbacks=[callback])

end_vgg = time()

vgg_train_dur = end_vgg - start_vgg
plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(VGG_results.history['accuracy'])

plt.plot(VGG_results.history['val_accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'])

plt.title('Accuracy evolution')



plt.subplot(122)

plt.plot(VGG_results.history['loss'])

plt.plot(VGG_results.history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'])

plt.title('Loss evolution');
print('Classification report:')

VGG_pred = VGG_model.predict(Test_data)

print(classification_report(VGG_pred.round(),Test_label))
VGG_train_loss = VGG_results.history['loss']

VGG_val_loss = VGG_results.history['val_loss']

VGG_train_acc = VGG_results.history['accuracy']

VGG_val_acc = VGG_results.history['val_accuracy']



VGG_rep = classification_report(VGG_pred.round(),Test_label, output_dict=True)
del VGG_model

del VGG_results

gc.collect()
# Creating the inception module



def inception_module(layer_input, fb1, fb2_in, fb2_out, fb3_in, fb3_out, fb4_out):



    conv1 = Conv2D(fb1, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_input)



    conv3 = Conv2D(fb2_in, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_input)

    conv3 = Conv2D(fb2_out, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv3)



    conv5 = Conv2D(fb3_in, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_input)

    conv5 = Conv2D(fb3_out, (5,5), padding='same', activation='relu', kernel_initializer='he_normal')(conv5)



    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_input)

    pool = Conv2D(fb4_out, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(pool)



    layer_output = concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_output
visible = Input(shape=(256,256,1))



layer = Conv2D(32, (7,7), strides=2, padding='same', activation='relu', kernel_initializer='he_normal' )(visible)

layer = MaxPooling2D((3,3), strides=(2,2))(layer)

layer = Lambda(tf.nn.local_response_normalization)(layer)

layer = Conv2D(32, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' )(layer)

layer = MaxPooling2D((3,3), strides=(2,2))(layer)

layer = inception_module(layer, 32, 64, 64, 16, 32, 32)

layer = inception_module(layer, 32, 64, 64, 16, 32, 32)

layer = MaxPooling2D((3,3), strides=(2,2), padding='same')(layer)

layer = Conv2D(32, (3,3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal' )(layer)

layer = AveragePooling2D((3,3), padding='valid')(layer)

layer = Flatten()(layer)

layer = Dense(4, activation='relu', kernel_initializer='he_normal')(layer)

layer = Dense(1, activation='sigmoid')(layer)



Inception_model = Model(inputs=visible, outputs=layer, name='Inception_model')



Inception_model.summary()

plot_model(Inception_model, show_shapes=True, to_file='inception_module.png')
Inception_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

callback = ReduceLROnPlateau(monitor='val_loss', patience = 3, cooldown=0, verbose=1, factor=0.6, min_lr=0.000001)

start_inception = time()

results_inception = Inception_model.fit(

    x=Train_data,

    y=Train_label,

    batch_size=16,

    validation_data=(Val_data,Val_label),

    class_weight={0:12, 1:0.5},

    epochs=15, callbacks=[callback])

end_inception = time()

inception_train_dur = end_inception - start_inception
plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(results_inception.history['accuracy'])

plt.plot(results_inception.history['val_accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'])

plt.title('Accuracy evolution')



plt.subplot(122)

plt.plot(results_inception.history['loss'])

plt.plot(results_inception.history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'])

plt.title('Loss evolution');
print('Classification report:')

Inception_pred = Inception_model.predict(Test_data)

print(classification_report(Inception_pred.round(),Test_label))
Inception_train_loss = results_inception.history['loss']

Inception_val_loss = results_inception.history['val_loss']

Inception_train_acc = results_inception.history['accuracy']

Inception_val_acc = results_inception.history['val_accuracy']



Inception_rep = classification_report(Inception_pred.round(),Test_label, output_dict=True)
del Inception_model

del results_inception

gc.collect()
# function that will use the identity if possible, otherwise a projection of the number of filters

# in the input does not match the n_filters argument.



def residual_module(layer_in, n_filters):

    

    merge_input = layer_in

    if layer_in.shape[-1] != n_filters[2]:

        merge_input = Conv2D(n_filters[2], (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

        

    conv1 = Conv2D(n_filters[0], (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    conv2 = Conv2D(n_filters[1], (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(conv1)

    conv3 = Conv2D(n_filters[2], (1,1), padding='same', activation='linear', kernel_initializer='he_normal')(conv2)

    

    layer_out = add([conv3, merge_input])

    layer_out = Activation('relu')(layer_out)

    

    return layer_out
visible = Input(shape=(256,256,1))



layer = Conv2D(32, (7,7), strides=(2,2) ,padding='same', activation='relu', kernel_initializer='he_normal')(visible)

layer = MaxPooling2D((2,2), strides=2 ,padding='same')(layer)

layer = residual_module(layer, [8,8,16])

layer = residual_module(layer, [8,8,16])

layer = BatchNormalization()(layer)

layer = residual_module(layer, [8,8,16])

layer = residual_module(layer, [8,8,16])

layer = BatchNormalization()(layer)

layer = residual_module(layer, [16,16,32])

layer = residual_module(layer, [16,16,32])

layer = BatchNormalization()(layer)

layer = residual_module(layer, [32,32,64])

layer = residual_module(layer, [32,32,64])

layer = AveragePooling2D((3,3), padding='valid')(layer)

layer = Flatten()(layer)

layer = Dense(16, activation='relu')(layer)

layer = Dropout(0.1)(layer)

layer = Dense(1, activation='sigmoid')(layer)



ResNet_model = Model(inputs=visible, outputs=layer, name='ResNet_model')



ResNet_model.summary()
plot_model(ResNet_model, show_shapes=True, to_file='ResNet_module.png')
ResNet_model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

callback = ReduceLROnPlateau(monitor='val_loss', patience = 2, cooldown=0, verbose=1, factor=0.6, min_lr=0.000001)

start_ResNet = time()

results_ResNet = ResNet_model.fit(

    x=Train_data,

    y=Train_label,

    batch_size=16,

    validation_data=(Val_data,Val_label),

    class_weight={0:20, 1:0.5},

    epochs=15, callbacks=[callback])

end_ResNet = time()

ResNet_train_dur = end_ResNet - start_ResNet
plt.figure(figsize=(12,5))

plt.subplot(121)

plt.plot(results_ResNet.history['accuracy'])

plt.plot(results_ResNet.history['val_accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Val'])

plt.title('Accuracy evolution')



plt.subplot(122)

plt.plot(results_ResNet.history['loss'])

plt.plot(results_ResNet.history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Val'])

plt.title('Loss evolution');
print('Classification report:')

ResNet_pred = ResNet_model.predict(Test_data)

print(classification_report(ResNet_pred.round(),Test_label))
ResNet_train_loss = results_ResNet.history['loss']

ResNet_val_loss = results_ResNet.history['val_loss']

ResNet_train_acc = results_ResNet.history['accuracy']

ResNet_val_acc = results_ResNet.history['val_accuracy']



ResNet_rep = classification_report(ResNet_pred.round(),Test_label, output_dict=True)
del ResNet_model

del results_ResNet

gc.collect()
plt.figure(figsize=(13,5))



plt.subplot(121)

plt.plot(VGG_val_acc)

plt.plot(Inception_val_acc)

plt.plot(ResNet_val_acc)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['VGG','Inception','ResNet'])

plt.title('Validation accuracy evolution')

plt.grid(True)





plt.subplot(122)

plt.plot(VGG_val_loss)

plt.plot(Inception_val_loss)

plt.plot(ResNet_val_loss)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['VGG','Inception','ResNet'])

plt.title('Validation loss evolution')

plt.grid(True);
# Function to plot the confusion matrix of the trained model.



def plot_cm(mat,y_ture):

    df_cm = pd.DataFrame(mat, columns=np.unique(y_ture), index = np.unique(y_ture))

    df_cm.index.name = 'True Label'

    df_cm.columns.name = 'Predicted Label'

    sb.heatmap(df_cm, cmap="Blues", cbar=False, annot=True,annot_kws={"size": 10})

    plt.yticks(fontsize=10)

    plt.xticks(fontsize=10)
plt.figure(figsize=(11,4))

plt.subplot(131)

plot_cm(confusion_matrix(VGG_pred.round(),Test_label, normalize='true'), Test_label)

plt.title('VGG model confusion matrix')

plt.subplot(132)

plot_cm(confusion_matrix(Inception_pred.round(),Test_label, normalize='true'), Test_label)

plt.title('Inception model confusion matrix')

plt.subplot(133)

plot_cm(confusion_matrix(ResNet_pred.round(),Test_label, normalize='true'), Test_label)

plt.title('ResNet model confusion matrix')

plt.tight_layout()
vgg_acc = round(VGG_rep['accuracy'],2)

inception_acc = round(Inception_rep['accuracy'],2)

resnet_acc = round(ResNet_rep['accuracy'],2)



vgg_normal_f1 = round(VGG_rep['0.0']['f1-score'],2)

inception_normal_f1 = round(Inception_rep['0.0']['f1-score'],2)

resnet_normal_f1 = round(ResNet_rep['0.0']['f1-score'],2)



vgg_pneu_f1 = round(VGG_rep['1.0']['f1-score'],2)

inception_pneu_f1 = round(Inception_rep['1.0']['f1-score'],2)

resnet_pneu_f1 = round(ResNet_rep['1.0']['f1-score'],2)





print('Comparaison of model results:')



result_table=[['Model','Training epochs' ,'Training duration (min)', 'Test accuracy', '(Normal) F1-score', '(Pneumonia) F1-score'],

             ['VGG model', 20, round(vgg_train_dur/60), vgg_acc, vgg_normal_f1, vgg_pneu_f1],

             ['Inception model', 15,round(inception_train_dur/60), inception_acc, inception_normal_f1, inception_normal_f1],

             ['ResNet model ', 15,round(ResNet_train_dur/60), resnet_acc, resnet_normal_f1, resnet_pneu_f1]]



display(HTML(tabulate.tabulate(result_table, colalign=("center",)*6, tablefmt='html')))