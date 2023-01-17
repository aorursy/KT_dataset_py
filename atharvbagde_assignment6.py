import datetime as dt

import matplotlib.pyplot as plt

import matplotlib.image as img

import numpy as np

import os

import pandas as pd

import seaborn as sns

from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications import resnet50

from tensorflow.keras.applications import vgg16

from tensorflow.keras.applications import inception_v3

from keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU

from tensorflow.keras.activations import swish

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.metrics import accuracy_score, confusion_matrix

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import random

import imgaug as ia

import imgaug.augmenters as iaa

from keras.utils import layer_utils

from tensorflow.keras.utils import to_categorical

from statistics import mean

import math

import cv2

from tensorflow import keras

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
!mkdir ~/.keras

!mkdir ~/.keras/models

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/

!mkdir -p /kaggle/working/plant-seedlings-classification/train
base_directory='../input/plant-seedlings-classification/'

train_dir=os.path.join(base_directory,'train')

test_dir=os.path.join(base_directory,'test')

save_dir = "/kaggle/working/plant-seedlings-classification/train"

Classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']



n_classes = len(Classes)



Classes
def augmentation_classes(Classes,train_dir):

    size_dict={}

    for i,classes in enumerate(Classes):

         size_dict[Classes[i]]=len(os.listdir(os.path.join(train_dir, classes)))

    print('Sample sizes of different classes are\n\n',size_dict)

    values_list=list(size_dict.values())

    ideal_samples=math.ceil(mean(values_list)*1.1)

    required_aug=[]

    for i,j in enumerate(size_dict):

        if size_dict[j]<ideal_samples:

            required_aug.append(j)

    print('\n\nMinority classes requiring augmentations are\n',required_aug)

    return required_aug,ideal_samples,size_dict

            





   

def image_augmentation(raw_images):

    

    

    seq=iaa.Sequential([iaa.Fliplr(0.5),

                        iaa.Flipud(0.3),

                        iaa.LinearContrast((0.75, 1.5)),

                        iaa.Crop(percent=(0, 0.2)),

                        iaa.Affine(rotate=(-45, 45)),

                        iaa.GaussianBlur(sigma=(0.0, 3.0))

                        ])

    image_aug=seq(images=raw_images)

    return image_aug

    
def preprocessing(img_path):

    image = cv2.resize(cv2.imread(img_path), (224,224), interpolation = cv2.INTER_NEAREST)

    return image
def get_training_data( model):

        

    if model == "resnet50":

        datagen = ImageDataGenerator(preprocessing_function = resnet50.preprocess_input, validation_split=0.15)

    elif model == "inceptionV3":

        datagen = ImageDataGenerator(preprocessing_function = inception_v3.preprocess_input, validation_split=0.15)

    elif model == 'vgg16':

        datagen = ImageDataGenerator(preprocessing_function = vgg16.preprocess_input, validation_split=0.15)



    train_data_den = datagen.flow_from_directory(

            directory= os.path.join(save_dir),

            class_mode = "categorical",

            batch_size=32,

            shuffle=True,

            subset='training'

        )

        

    val_data_gen = datagen.flow_from_directory(

            directory= os.path.join(save_dir),

            class_mode = 'categorical',

            batch_size=32,

            shuffle=False,

            subset='validation'

        )



    return train_data_den, val_data_gen
def augment_and_store_data(Classes,train_dir,save_dir,required_aug,ideal_samples,size_dict):

    for i,sample_class in enumerate(Classes):

        try:

            os.mkdir(os.path.join(save_dir,sample_class))

        except FileExistsError:

            pass

        img_list=[]

        for img_loc in os.listdir(os.path.join(train_dir,sample_class)):

            image = preprocessing(os.path.join(train_dir, sample_class, img_loc))

            img_list.append(image)

        if sample_class in required_aug:

            aug_img= image_augmentation(img_list)

            req_img=random.sample(aug_img,(ideal_samples-size_dict[sample_class]))

            img_list.extend(req_img)

        for image_number, image in enumerate(img_list):

            cv2.imwrite(os.path.join(save_dir, sample_class, "{}.png".format(image_number + 1)), image)

def model_prep(model_arch,monitor,lr_patience,early_stop_patience,min_lr):

    checkpoint = ModelCheckpoint(filepath=os.path.join('/kaggle/working/',model_arch,'.h5'), monitor=monitor, mode='min', save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=1, min_lr=min_lr)

    early_stop = EarlyStopping(monitor=monitor, min_delta=0, patience=5, verbose=1, mode='min', restore_best_weights=True)

    callbacks=[checkpoint,reduce_lr,early_stop]

    train_gen, val_gen = get_training_data(model = model_arch)

    

    return callbacks,train_gen,val_gen

    
a={}

for i,classes in enumerate(Classes):

     a[i]=len(os.listdir(os.path.join(train_dir, classes)))

a=pd.DataFrame(a.items(),columns=['index','no.of samples'],index=a.keys())

a['no.of samples'].plot(kind='bar')

sample_size=a['no.of samples']
required_aug,ideal_samples,size_dict = augmentation_classes(Classes,train_dir)
augment_and_store_data(Classes,train_dir,save_dir,required_aug,ideal_samples,size_dict)
c={}

for i,classes in enumerate(Classes):

     c[i]=len(os.listdir(os.path.join(save_dir, classes)))

c=pd.DataFrame(c.items(),columns=['index','no.of samples'],index=c.keys())

c['no.of samples'].plot(kind='bar')

sample_size=a['no.of samples']
callbacks,train_gen,val_gen=model_prep('resnet50','val_loss',2,5,0.000001)
resnet50_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

x = resnet50_model.output

x = Dropout(0.6)(x)

x = Dense(256)(x)

x = BatchNormalization()(x)

x = swish(x)

pred = Dense(12, activation='softmax')(x)

final_model = Model(inputs = resnet50_model.input, outputs = pred)



for layer in resnet50_model.layers[0:-9]:

    layer.trainable = False

    

final_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
resnet50_model.summary()
hist_resnet50 = final_model.fit_generator(train_gen,

                      steps_per_epoch = 155,

                      validation_data = val_gen,

                      epochs = 50,

                      verbose = 1,

                      callbacks = callbacks)
plt.figure(1, figsize=(10,13 ))

for i in range(len(hist_resnet50.history['val_loss'])):

  plt.subplot(2,1,1)

  plt.title('Cross Entropy Loss')

  plt.plot(hist_resnet50.history['loss'], color='blue', label='Train')

  plt.plot(hist_resnet50.history['val_loss'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)



  plt.subplot(2,1,2)

  plt.title('Classification Accuracy')

  plt.plot(hist_resnet50.history['accuracy'], color='blue', label='Train')

  plt.plot(hist_resnet50.history['val_accuracy'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)
final_model.save('/kaggle/working/resnet50_saved_model')
callbacks,train_gen,val_gen=model_prep('vgg16','val_loss',2,5,0.000001)
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

x = vgg_model.output

x = Dropout(0.6)(x)

x = Dense(256)(x)

x = BatchNormalization()(x)

x = swish(x)

pred = Dense(12, activation='softmax')(x)

final_vgg_model = Model(inputs = vgg_model.input, outputs = pred)



for layer in vgg_model.layers[0:-4]:

    layer.trainable = False

    

final_vgg_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
vgg_model.summary()
history_vgg16 = final_vgg_model.fit_generator(train_gen,

                      steps_per_epoch = 155,

                      validation_data = val_gen,

                      epochs = 50,

                      verbose = 1,

                      callbacks = callbacks)
plt.figure(1, figsize=(14,8 ))

for i in range(len(history_vgg16.history['val_loss'])):

  plt.subplot(2,1,1)

  plt.title('Cross Entropy Loss')

  plt.plot(history_vgg16.history['loss'], color='blue', label='Train')

  plt.plot(history_vgg16.history['val_loss'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)



  plt.subplot(2,1,2)

  plt.title('Classification Accuracy')

  plt.plot(history_vgg16.history['accuracy'], color='blue', label='Train')

  plt.plot(history_vgg16.history['val_accuracy'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)
final_vgg_model.save('/kaggle/working/vgg_saved_model')
callbacks,train_gen,val_gen=model_prep('inceptionV3','val_loss',2,5,0.000001)
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

x = inception_model.output

x = Dropout(0.5)(x)

x = Dense(512)(x)

x = BatchNormalization()(x)

x = swish(x)

x = Dropout(0.5)(x)

pred = Dense(12, activation='softmax')(x)

final_inception_model = Model(inputs = inception_model.input, outputs = pred)



for layer in inception_model.layers[0:-22]:

    layer.trainable = False

    

final_inception_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
inception_model.summary()
history_inception_v3 = final_inception_model.fit_generator(train_gen,

                      steps_per_epoch = 155,

                      validation_data = val_gen,

                      epochs = 50,

                      verbose = 1,

                      callbacks = callbacks)
plt.figure(1, figsize=(14,8 ))

for i in range(len(history_inception_v3.history['val_loss'])):

  plt.subplot(2,1,1)

  plt.title('Cross Entropy Loss')

  plt.plot(history_inception_v3.history['loss'], color='blue', label='Train')

  plt.plot(history_inception_v3.history['val_loss'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)



  plt.subplot(2,1,2)

  plt.title('Classification Accuracy')

  plt.plot(history_inception_v3.history['accuracy'], color='blue', label='Train')

  plt.plot(history_inception_v3.history['val_accuracy'], color='orange', label='validation')

  plt.legend(['Train', 'validation'], fontsize = 14)
final_inception_model.save('/kaggle/working/inceptionV3_saved_model')
incep_model = keras.models.load_model('/kaggle/working/inceptionV3_saved_model')
callbacks,train_gen,val_gen=model_prep('inceptionV3','val_loss',2,5,0.000001)

incep_model.evaluate_generator(generator=val_gen)
predictions = incep_model.predict_generator(val_gen,28)
y_pred = np.argmax(predictions, axis=1)

cf_matrix = confusion_matrix(val_gen.classes, y_pred)

print('Classification Report')

print(classification_report(val_gen.classes, y_pred, target_names=Classes))

plt.figure(figsize=(20,20))

sns.heatmap(cf_matrix, annot=True, xticklabels=Classes, yticklabels=Classes, cmap='Blues')
resnet_model = keras.models.load_model('/kaggle/working/resnet50_saved_model')
callbacks,train_gen,val_gen=model_prep('resnet50','val_loss',2,5,0.000001)

resnet_model.evaluate_generator(generator=val_gen)
predictions = resnet_model.predict_generator(val_gen,28)
y_pred = np.argmax(predictions, axis=1)

cf_matrix = confusion_matrix(val_gen.classes, y_pred)

print('Classification Report')

print(classification_report(val_gen.classes, y_pred, target_names=Classes))

plt.figure(figsize=(20,20))

sns.heatmap(cf_matrix, annot=True, xticklabels=Classes, yticklabels=Classes, cmap='Blues')
callbacks,train_gen,val_gen=model_prep('vgg16','val_loss',2,5,0.000001)

vgg_model = keras.models.load_model('/kaggle/working/vgg_saved_model')

vgg_model.evaluate_generator(generator=val_gen)
predictions = vgg_model.predict_generator(val_gen,28)

y_pred = np.argmax(predictions, axis=1)

cf_matrix = confusion_matrix(val_gen.classes, y_pred)

print('Classification Report')

print(classification_report(val_gen.classes, y_pred, target_names=Classes))
plt.figure(figsize=(20,20))

sns.heatmap(cf_matrix, annot=True, xticklabels=Classes, yticklabels=Classes, cmap='Blues')


!zip-folder --auto-root --outfile /kaggle/working.zip /kaggle/working 