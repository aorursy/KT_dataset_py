import tensorflow as tf
import keras

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_score

from scipy import stats

import seaborn as sns

import skimage
from skimage.transform import rotate

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, vgg16, resnet50, inception_v3, xception, DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import numpy as np
import os
import cv2
import pandas as pd
# import imutils
import random
from PIL import Image
import matplotlib.pyplot as plt
DATA_PATH = "../input/super-ai-image-classification/"

TRAIN_PATH = DATA_PATH + "train/train/"
TEST_PATH = DATA_PATH + "val/val/"

df = pd.read_csv(TRAIN_PATH + "train.csv")
df.index = df['id']

x_resolution = 150
y_resolution = 150
batch_size = 64
batch_size_generator = 32
epoch = 20
earlystopping = 30
# Without Drop images
def load_image():
    label = []
    img_path = os.listdir(TRAIN_PATH + "images")
    train_img = []

    for image in img_path:
        train_image = cv2.imread(TRAIN_PATH + "images/"+image)/255.
        train_image = cv2.resize(train_image, (x_resolution, y_resolution))
        
        train_img.append(train_image)
        label.append(df.category[image])
    
    print(np.array(train_img).shape)
    print(np.array(label).shape)
    
    return np.array(train_img), np.array(label)

# Drop images
def load_img(df):
    label = []
    sum_img = []
    train_img = []

    count = 0
    for image in df.id:
        train_img1 = cv2.imread(TRAIN_PATH + "images/"+image)
        sum_img.append([int(np.sum(train_img1)),image])

    sum_img = sorted(sum_img)
    memory = 0
    same = 0
    for i in range(len(sum_img)):
        if sum_img[i][0] == memory:
            same += 1
            df = df.drop(sum_img[i][1])
        memory = sum_img[i][0]

    count = 0
    for image in df.id:
        train_img1 = cv2.imread(TRAIN_PATH + "images/"+image)/255.
        train_img1 = cv2.resize(train_img1, (x_resolution, y_resolution))/255.
#         print(f"Image {count}: ",train_img1.shape,df.category[image])
        train_img.append(train_img1)
        label.append(df.category[image])
#         print(f"Load Image {count}: Complete!")
        count += 1

    print(same)
    return np.array(train_img), np.array(label)

# X, Y = load_img(df)
X, Y = load_image()
X.shape, Y.shape
# X, Y = load_image()
# aug = ImageDataGenerator(
#         rotation_range=15,
#         width_shift_range=0.15,
#         height_shift_range=0.15,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True
#         )

# fig = plt.figure(tight_layout='auto', figsize=(10, 7))
# for num, i in enumerate(aug.flow(X)):
#     fig.add_subplot(331+int(num))
#     plt.imshow(i[0])
#     if num == 8:
#         break
# plt.show()
with tf.device('/device:GPU:0'):        
    def get_f1(y_true, y_pred): #taken from old keras source code
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
        return f1_val
    
    def get_model():
        inputs = Input(shape=(x_resolution, y_resolution, 3))
        
        x = inputs
        basemodel = InceptionV3(include_top=False, input_shape=(x_resolution, y_resolution, 3), weights='imagenet')
        
        for layer in basemodel.layers:
            layer.trainable = False
            
#         basemodel.trainable = False

        x = basemodel(x)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(1)(x)
        outputs = Activation('sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model
# x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

kfold = KFold(n_splits=10, random_state=42, shuffle=False)

# train_generator = aug.flow(x_train, y_train, seed=42, batch_size=batch_size_generator)
# valid_generator = aug.flow(x_valid, y_valid, seed=42, batch_size=batch_size_generator)

model = get_model()
# model_generator = get_model()
np.unique(Y, return_counts=True)
folder = 'drop_1'
best_model_filename = './model_'+folder+'.h5'

EarlyStopper = EarlyStopping(patience=earlystopping, verbose=1, monitor='val_accuracy', mode='max')
Csv_logger = CSVLogger('./save.csv', append=True, separator=';')
Checkpoint = ModelCheckpoint(best_model_filename, verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
# model.fit(x_train, y_train, 
#          validation_data=(x_valid, y_valid),
#          batch_size=batch_size,
#          epochs=epoch,
#          verbose=1,
#          callbacks=[Csv_logger, Checkpoint, EarlyStopper])
val_acc = []
for count, (train_index, valid_index) in enumerate(kfold.split(X)):
#     x_train = X[train_index]
#     y_train = Y[train_index]
#     x_valid = X[valid_index]
#     y_valid = Y[valid_index]
    
    model.fit(X[train_index], Y[train_index], 
         validation_data=(X[valid_index], Y[valid_index]),
         batch_size=batch_size,
         epochs=epoch,
         verbose=1
        )
    print(count+1)
    print('========= Genearator Model =========')
    results_train = model.evaluate(X[train_index], Y[train_index])
    results_valid = model.evaluate(X[valid_index], Y[valid_index])
    print('Train loss :', results_train[0])
    print('Train accuracy :', results_train[1])
    print('Valid loss :', results_valid[0])
    print('Valid accuracy :', results_valid[1])
    val_acc.append(results_valid[1])
    print('\n\n')
    
model.save(best_model_filename)
for num, i in enumerate(val_acc):
    print('Fold :', num+1)
    print('Accuracy :', i*100, '%')
    print('=========================================')
    
print('========= Last Genearator Model =========')
results = model.evaluate(X, Y)
print('Loss :', results[0])
print('Accuracy :', results[1]*100, '%')

# print('========= Last Model =========')
# results_train = model.evaluate(x_train, y_train)
# results_valid = model.evaluate(x_valid, y_valid)
# print('Train loss :', results_train[0])
# print('Train accuracy :', results_train[1])
# print('Valid loss :', results_valid[0])
# print('Valid accuracy :', results_valid[1])

# best_model = load_model(best_model_filename)
# print('\n\n========= Best Model =========')
# results__train = best_model.evaluate(x_train, y_train)
# results__valid = best_model.evaluate(x_valid, y_valid)
# print('Train loss :', results__train[0])
# print('Train accuracy :', results__train[1])
# print('Valid loss :', results__valid[0])
# print('Valid accuracy :', results__valid[1])
# best_model_generator_filename = './best_model_generator.h5'

# EarlyStopper = EarlyStopping(patience=earlystopping, verbose=1, monitor='val_get_f1', mode='max')
# Csv_logger = CSVLogger('./save_generator.csv', append=True, separator=';')
# Checkpoint = ModelCheckpoint(best_model_generator_filename, verbose=1, monitor='val_get_f1', save_best_only=True, mode='max')
# for count, (train_index, valid_index) in enumerate(kfold.split(X)):
#     train_generator = aug.flow(X[train_index], Y[train_index], seed=42, batch_size=batch_size_generator)
#     valid_generator = aug.flow(X[valid_index], Y[valid_index], seed=42, batch_size=batch_size_generator)
#     model_generator.fit(train_generator, 
#          validation_data=valid_generator,
#          steps_per_epoch=len(X[train_index])//batch_size,
#          epochs=epoch,
#          verbose=0,
#          callbacks=[EarlyStopper]
#         )
#     print(count+1)
#     print('========= Genearator Model =========')
#     results_train = model_generator[count].evaluate(X[train_index], Y[train_index])
#     results_valid = model_generator[count].evaluate(X[valid_index], Y[valid_index])
#     print('Train loss :', results_train[0])
#     print('Train accuracy :', results_train[1])
#     print('Valid loss :', results_valid[0])
#     print('Valid accuracy :', results_valid[1])
#     print('\n\n')
    
# # model_generator.save('model.h5')

# print('========= Last Genearator Model =========')
# results = model_generator.evaluate(X, Y)
# print('Loss :', results[0])
# print('Accuracy :', results[1])
# model_generator.fit(train_generator, 
#          validation_data=valid_generator,
#          batch_size=len(x_train)//batch_size,
#          epochs=epoch,
#          verbose=1,
#          callbacks=[Csv_logger, Checkpoint, EarlyStopper])
# print('========= Last Genearator Model =========')
# results_train = model_generator.evaluate(x_train, y_train)
# results_valid = model_generator.evaluate(x_valid, y_valid)
# print('Train loss :', results_train[0])
# print('Train accuracy :', results_train[1])
# print('Train f1 :', results_train[2])
# print('Valid loss :', results_valid[0])
# print('Valid accuracy :', results_valid[1])
# print('Valid f1 :', results_valid[2])

# best_model_generator = load_model(best_model_generator_filename)
# print('\n\n========= Best Generator Model =========')
# results__train = best_model_generator.evaluate(x_train, y_train)
# results__valid = best_model_generator.evaluate(x_valid, y_valid)
# print('Train loss :', results__train[0])
# print('Train accuracy :', results__train[1])
# print('Train f1 :', results__train[2])
# print('Valid loss :', results__valid[0])
# print('Valid accuracy :', results__valid[1])
# print('Valid f1 :', results__valid[2])
# print('========= Last Genearator Model =========')
# results_train = model_generator.evaluate(train_generator)
# results_valid = model_generator.evaluate(valid_generator)
# print('Train loss :', results_train[0])
# print('Train accuracy :', results_train[1])
# print('Train f1 :', results_train[2])
# print('Valid loss :', results_valid[0])
# print('Valid accuracy :', results_valid[1])
# print('Valid f1 :', results_valid[2])

# print('\n\n========= Best Generator Model =========')
# results__train = best_model_generator.evaluate(train_generator)
# results__valid = best_model_generator.evaluate(valid_generator)
# print('Train loss :', results__train[0])
# print('Train accuracy :', results__train[1])
# print('Train f1 :', results__train[2])
# print('Valid loss :', results__valid[0])
# print('Valid accuracy :', results__valid[1])
# print('Valid f1 :', results__valid[2])
def load_img_test():
    img_path = os.listdir(TEST_PATH + "images")
    test_img = np.zeros((len(img_path), x_resolution, y_resolution, 3)).astype('float')

    count = 0
    for image in img_path:
        test_img1 = cv2.imread(TEST_PATH + "images/"+image)/255.
        test_img1 = cv2.resize(test_img1, (x_resolution, y_resolution))
        #print(f"Image {count}: ",test_img1.shape)
        test_img[count] = test_img1
        #print(f"Load Image {count}: Complete!")
        count += 1

    return test_img

test_data = pd.read_csv(TEST_PATH + "val.csv")
test_img = load_img_test()
test_img = np.array(test_img)
print(test_img.shape)
best_model = load_model(best_model_filename)
pred = best_model.predict(test_img)
label = []
for value in range(len(pred)):
    if pred[value] >= 0.5:
        label.append(1)
    else:
        label.append(0)

data = {'id':os.listdir(TEST_PATH + "images"), 'category':label}
submission_df = pd.DataFrame(data)
submission_df
submission_df.to_csv('submission_091020_'+folder+'.csv',index=False)