import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import time
from sklearn.preprocessing  import LabelEncoder 
from keras.utils.np_utils import to_categorical
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read data script
train = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
train_rules = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
test_rules = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
submit = pd.read_csv('../input/thai-mnist-classification/submit.csv')
#Data Augmentation
datagen = ImageDataGenerator(
          rotation_range=10,  
          zoom_range = 0.10,  
          width_shift_range=0.1, 
          height_shift_range=0.1,
        rescale=1.0/255,
        validation_split=0.2)
#Prepare train and validation
train['category'] = train['category'].astype('str') # requires target in string format
batch_size = 64
IMG_SIZE = 256
#batch_size = 16 * tpu_strategy.num_replicas_in_sync



train_generator_df = datagen.flow_from_dataframe(dataframe=train, 
                                              directory='../input/thai-mnist-classification/train/',
                                              x_col="id", 
                                              y_col="category", 
                                              subset="training",
                                              class_mode="categorical", 
                                              target_size=(IMG_SIZE, IMG_SIZE), 
                                              batch_size=batch_size,
                                              seed=0)
valid_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory='../input/thai-mnist-classification/train/',
                                              x_col="id", 
                                              y_col="category", 
                                              subset="validation",
                                              class_mode="categorical", 
                                              target_size=(IMG_SIZE, IMG_SIZE), 
                                              batch_size=batch_size,
                                              seed=0)
#Define models

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7,vgg16,vgg19
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_model_EfficientNetB0(IMG_SIZE,NUM_CLASSES):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def build_model_EfficientNetB7(IMG_SIZE,NUM_CLASSES):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = img_augmentation(inputs)
    model = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB7")
    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def alexnet(IMG_SIZE,num_classes):
    # Initializing the CNN
    model = Sequential()

    # Convolution Step 1
    model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))

    # Max Pooling Step 1
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())

    # Convolution Step 2
    model.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

    # Max Pooling Step 2
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Convolution Step 3
    model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    model.add(BatchNormalization())

    # Convolution Step 4
    model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    model.add(BatchNormalization())

    # Convolution Step 5
    model.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

    # Max Pooling Step 3
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())

    # Flattening Step
    model.add(Flatten())

    # Full Connection Step
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units = 1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(units = num_classes, activation = 'softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
def build_model_vgg16():
    base = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    for l in base.layers:
        l.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(base.input, x,name='vgg16')
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def build_model_vgg19():
    base = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    for l in base.layers:
        l.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units = 128, activation = 'relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(base.input, x,name='vgg19')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
#Build 15 models [Result is predicted by 13 VGG16 and 2 VGG19]
nets = 15
model = [0] *nets
for i in range(nets):
    #model[i] = alexnet(IMG_SIZE,10)
    #model[i] = build_model_EfficientNetB0(IMG_SIZE,10)
    #model[i] = build_model_EfficientNetB7(IMG_SIZE,10)
    model[i] = build_model_vgg16()
    #model[i] = build_model_vgg19()
model[0].summary()
# Learning strategy
annealer = LearningRateScheduler(lambda x: 1e-2 * 0.97 ** x)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15,verbose=1)
# Train models
start_time = time.time()
STEP_SIZE_TRAIN=train_generator_df.n//train_generator_df.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = [0] * nets
epochs = 50
for j in range(nets):
    history[j] = model[j].fit_generator(
        train_generator_df,
        epochs = epochs, 
        steps_per_epoch = STEP_SIZE_TRAIN,  
        validation_data = valid_generator,
        validation_steps=STEP_SIZE_VALID, 
        callbacks=[annealer,early_stop], 
        verbose=1)
    #print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    model[j].save('./vgg16_model'+str(j)+'.hdf5') #Save model for future use
train_rules.isnull().sum(axis = 0)
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
#load trained models
from tensorflow.keras.models import load_model

modelPaths = list(list_files('../input/trained-vgg-models/'))
trained_models = [0] *len(modelPaths)

for i,modelPath in enumerate(modelPaths):
    trained_models[i] = load_model(modelPath)
len(trained_models)
test_rules.isna().sum()
# Ensemble prediction
def ENSEMBLE_Models_Predict(models,t_generator,verbose=True):
    results = np.zeros((t_generator.n,10)) 
    for j in range(len(models)):
        results = results + models[j].predict_generator(t_generator, verbose=verbose) #vote by sum
    results = np.argmax(results,axis = 1)
    return results #numpy

def prepare_feature_for_predict(feature_name):
    df = pd.DataFrame()     
    df[feature_name] = test_rules[feature_name].dropna()
    return df
feature1 = prepare_feature_for_predict('feature1')
feature2 = prepare_feature_for_predict('feature2')
feature3 = prepare_feature_for_predict('feature3')
feature1

feature2
feature3
feature_datagen = ImageDataGenerator(rescale=1.0/255)
feature1_generator = feature_datagen.flow_from_dataframe(dataframe=feature1, directory='../input/thai-mnist-classification/test/',x_col="feature1",y_col=None,shuffle=False,class_mode=None)
feature2_generator = feature_datagen.flow_from_dataframe(dataframe=feature2, directory='../input/thai-mnist-classification/test/',x_col="feature2",y_col=None,shuffle=False,class_mode=None)
feature3_generator = feature_datagen.flow_from_dataframe(dataframe=feature3, directory='../input/thai-mnist-classification/test/',x_col="feature3",y_col=None,shuffle=False,class_mode=None)
predict_feature1 = ENSEMBLE_Models_Predict(trained_models,feature1_generator,verbose=True)
p_f1 = pd.concat([pd.DataFrame(predict_feature1,columns=['p_f1']),feature1.reset_index()['feature1']], axis=1)
p_f1.to_csv('p_f1.csv')
predict_feature2 = ENSEMBLE_Models_Predict(trained_models,feature2_generator,verbose=True)
p_f2 = pd.concat([pd.DataFrame(predict_feature2,columns=['p_f2']),feature2.reset_index()['feature2']], axis=1)
p_f2.to_csv('p_f2.csv')
predict_feature3 = ENSEMBLE_Models_Predict(trained_models,feature3_generator,verbose=True)
p_f3 = pd.concat([pd.DataFrame(predict_feature3,columns=['p_f3']),feature3.reset_index()['feature3']], axis=1)
p_f3.to_csv('p_f3.csv')
p_f1 = pd.read_csv('../input/predict-features/p_f1.csv')
p_f2 = pd.read_csv('../input/predict-features/p_f2.csv')
p_f3 = pd.read_csv('../input/predict-features/p_f3.csv')
p_f1
def find_relate(x,predict_df,featureName,PredictFeatureName):
    try:
        n = predict_df[predict_df[featureName]==x]
        return n[PredictFeatureName].values[0]
    except:
        return -1
test_set = test_rules.fillna(-1)
test_set
test_set['feauture1_trans'] = test_set.apply(lambda x:  find_relate(x['feature1'],p_f1,'feature1','p_f1') , axis=1)
test_set['feauture2_trans'] = test_set.apply(lambda x:  find_relate(x['feature2'],p_f2,'feature2','p_f2') , axis=1)
test_set['feauture3_trans'] = test_set.apply(lambda x:  find_relate(x['feature3'],p_f3,'feature3','p_f3') , axis=1)
test_set
test_set.to_csv('test_data.csv')
def rules_based_prediction(a,b,c):
    if(a==0):
        return b*c
    if(a == 1):
        return abs(b-c)
    if(a == 2):
        return (b+c)*abs(b-c)
    if(a==3):
        return abs((c*(c +1) - b*(b-1))//2)
    if(a==4):
        return 50+b-c
    if(a==5):
        return min(b,c)
    if(a==6):
        return max(b,c)
    if(a==7):
        return ((b*c)%9)*11
    if(a==8):
        return (((b**2)+1)*(b) +(c)*(c+1))%99
    if(a==9):
        return 50+b
    return b+c
test_set["f_predict"] = [rules_based_prediction(row['feauture1_trans'],row['feauture2_trans'],row['feauture3_trans']) for index, row in test_set.iterrows()]
test_set.to_csv('test_data_with_predict_by_rule_base.csv')
test_set
submit['predict'] = submit.apply(lambda x:  find_relate(x['id'],test_set,'id','f_predict') , axis=1)
submit
submit.to_csv('submit1.csv',index=False)