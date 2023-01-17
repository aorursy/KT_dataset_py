import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle

import warnings
warnings.filterwarnings("ignore")
SIZE = 512
# Load dataset info
path_to_train = '../input/human-protein-atlas-image-classification/train/'
data = pd.read_csv('../input/human-protein-atlas-image-classification/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)   
                    if augument:
                        if not np.all(X_train_batch[i]['labels']==(0,25)):
                            #to balance the training data I don´t augument the labels 0 and 25
                            image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image_red_yellow_ch = PIL.ImageChops.blend(image_red_ch,image_yellow_ch,0.5)
        image = np.stack((
        np.array(image_green_ch), 
        np.array(image_red_yellow_ch), 
        np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]),interpolation = cv2.INTER_CUBIC)
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, GlobalAveragePooling2D,BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam, Adagrad, Adamax 
from keras import backend as K
import keras
from keras.models import Model

    
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = NASNetMobile(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape) 
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1056, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 10; batch_size = 16
checkpoint = ModelCheckpoint('../working/InceptionV3.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6)
callbacks_list = [checkpoint]

# split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.19, random_state=8)

# create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)

model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=28)



model.compile(loss='binary_crossentropy',
            optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0),
            metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=5, 
    verbose=1,
    callbacks=callbacks_list)
#model.compile(loss='binary_crossentropy',
#            optimizer=Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001),
#            metrics=['accuracy'])
#model.fit_generator(
#    train_generator,
#    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
#    validation_data=validation_generator,
#    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
#    epochs=, 
#    verbose=1,
#    callbacks=callbacks_list)
submit = pd.read_csv('../input/human-protein-atlas-image-classification/sample_submission.csv')
predicted = []
draw_predict = []
model.load_weights('../working/InceptionV3.h5')
for name in tqdm(submit['Id']):
    path = os.path.join('../input/human-protein-atlas-image-classification/test/', name)
    image = data_generator.load_image(path, (SIZE,SIZE,3))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv('submit20.csv', index=False)