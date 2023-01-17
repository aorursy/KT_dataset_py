import numpy as np 
import pandas as pd
from tensorflow import keras
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import DenseNet201
#from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import Xception
from sklearn.utils import class_weight
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
#import cv2
import tensorflow as tf
from keras.models import Sequential
#import efficientnet.keras as efn
# import tensorflow.keras.applications.ResNet101 as resnet101

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from plotly import express
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
#import lightgbm as lgb
#import catboost as ctb
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv',na_values=['unknown'])
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
DIR = '../input/resize-jpg-siimisic-melanoma-classification/300x300/train/'
#DIR = './300x300/train'
TestDIR = '../input/resize-jpg-siimisic-melanoma-classification/300x300/test/'
img = []
train_fk=[]
labels = []
format = '.jpg'

for i in train['image_name']:
    img.append(os.path.join(DIR,i)+format)
    
for i in train['target']:
    labels.append(str(i))
for i in train['image_name']:         #save images name in an array to use later for kfold cross validation
  train_fk.append(i+format)

#creating a no array for test set as well
test_data=[]
for i in range(test.shape[0]):
    test_data.append(TestDIR + test['image_name'].iloc[i]+format)
df_test=pd.DataFrame(test_data)
df_test.columns=['images']        #this one alsong with above line are creating a dataframe for test data which will be used for the submision task
class_weights = class_weight.compute_class_weight(
    'balanced',
    train['target'].unique(),
    train[['target']].to_numpy().reshape(-1)
)
weights = {i : class_weights[i] for i in range(2)}
print('benign weight: ',class_weights[0])
print('malignant weight: ',class_weights[1])
#this library allow user to use a customized datagenerator with which one can use augmentation implemented in albumentation library 
!pip install git+https://github.com/mjkvaak/ImageDataAugmentor
from ImageDataAugmentor.image_data_augmentor import *
import albumentations
   
AUGMENTATIONS = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.RandomContrast(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    #albumentations.RandomCrop(10,10,p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.Cutout(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),  #default rotations is 45degree + shift height and width with the limit amount of 0.0625
    albumentations.OneOf([
    albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
    ],p=1),
   # albumentations.GaussianBlur(p=0.05),
    albumentations.HueSaturationValue(p=0.5),
    #albumentations.RGBShift(p=0.5),
])
train_datagen = ImageDataAugmentor(
        rescale=1./255,
        augment = AUGMENTATIONS,
        preprocess_input=None)
        
valid_datagen = ImageDataAugmentor(rescale=1./255)

#train_generator = train_datagen.flow_from_dataframe(train_data,
#                                                x_col='image',
#                                                y_col='target',
#                                                target_size=(224,224),
#                                                batch_size=32,
 #                                               shuffle=True,
  #                                              class_mode='raw')
        
#validation_generator = valid_datagen.flow_from_dataframe(val_data,
 #                                               x_col='image',
  #                                              y_col='target',
   #                                             target_size=(224,224),
    #                                            batch_size=32,
     #                                           shuffle=True,
      #                                          class_mode='raw')
from tensorflow.python.keras import backend as K

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        #compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
submission=pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
from sklearn.model_selection import StratifiedKFold
n_split = 5
cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=47)
fold_count = 0
df_img = pd.DataFrame(train_fk,columns=['image'])
print(type(labels))
df_labels = pd.DataFrame(labels,columns=['target'])
train_data = pd.concat([df_img, df_labels], axis = 1) 
target = train_data[['target']]
#print(np.shape(target))
for train_idx, val_idx in cv.split(df_img, target ):
    fold_count += 1
    print("this is the fold number ",fold_count)
    training_data = train_data.iloc[train_idx]
    validation_data = train_data.iloc[val_idx]
    print("tr= ",training_data.shape[0])
    print("val= ",validation_data.shape[0])
  

    train_generator = train_datagen.flow_from_dataframe(training_data,
                                                        directory=DIR,
                                                        x_col='image',
                                                        y_col='target',
                                                        target_size=(300,300),
                                                        batch_size=32,
                                                        shuffle=True,
                                                        class_mode='binary')
        
    validation_generator = valid_datagen.flow_from_dataframe(validation_data,
                                                             directory=DIR,
                                                             x_col='image',
                                                             y_col='target',
                                                            target_size=(300,300),
                                                            batch_size=32,
                                                            shuffle=True,
                                                            class_mode='binary')

    base_model_alb =  ResNet50V2(include_top=False, weights='imagenet',input_shape=(300,300 , 3))
  
    n_layers = len(base_model_alb.layers)
    for layer in base_model_alb.layers[:n_layers - 15]: #freezing some layers
      layer.trainable = False
    for layer in base_model_alb.layers[n_layers - 15:]:
      layer.trainable = True
  
    x = base_model_alb.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(out)

    model_alb = Model(inputs=base_model_alb.input, outputs=predictions)


    from tensorflow.keras.optimizers import Adam
    def scheduler(epoch, lr):
        if epoch<4:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                 tf.keras.callbacks.EarlyStopping(patience=4)]
    model_alb.compile(loss=focal_loss(),
                  optimizer=Adam(lr=1e-5),
                  metrics=[tf.keras.metrics.AUC()])

    History = model_alb.fit(train_generator,
                             steps_per_epoch=training_data.shape[0]//32,
                             epochs=15,
                             validation_data=validation_generator,
                             validation_steps=validation_data.shape[0]//32,
                             shuffle=False,
                             callbacks=callbacks,
                             class_weight=weights
                             )
    
    n_layers = len(base_model_alb.layers)
    for layer in base_model_alb.layers[:n_layers - 20]:
        layer.trainable = False
    for layer in base_model_alb.layers[n_layers - 20:]:
        layer.trainable = True
        
    
    def scheduler(epoch, lr):
        if epoch<8:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                 tf.keras.callbacks.EarlyStopping(patience=4)]
    model_alb.compile(loss=focal_loss(),
                  optimizer=Adam(lr=1e-6),
                  metrics=[tf.keras.metrics.AUC()])
    
    History = model_alb.fit(train_generator,
                             steps_per_epoch=training_data.shape[0]//32,
                             epochs=25,
                             validation_data=validation_generator,
                             validation_steps=validation_data.shape[0]//32,
                             shuffle=False,
                             callbacks=callbacks,
                             class_weight=weights
                             )
    
    from tensorflow.keras.preprocessing import image
    import tensorflow.keras.applications.resnet_v2 as tf_res
    from tensorflow.keras.applications import xception
    target=[]
    for pat in df_test['images']:
        img_path = str(pat)
        img = image.load_img(img_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) 
        y = tf_res.preprocess_input(x)
 # z = xception.preprocess_input(x)
        prediction=model_alb.predict(y)
        target.append(prediction[0][0]/n_split)   #give 1/5th share for predicition in each fold
    submission['target']+=target

submission.to_csv('submission.csv', index=False)
submission.head()