import cv2
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.utils import class_weight, shuffle
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
WINDOW_SIZE = 299
IMAGE_SIZE  = 512
IMAGE_CHANNELS=3
NUM_CLASSES=28

path_to_train = '../input/human-protein-atlas-image-classification'
data = pd.read_csv(path_to_train+'/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

class data_generator:
    def __init__(self, it):
        self.it = it
    def __call__(self):
        return self.it

    def get_dataset(dataset_info, batch_size, shape, augument=True):
        gen = data_generator.create_train(dataset_info, batch_size, shape, augument)
        gen = data_generator(gen)
        types = (tf.float32, tf.float32)
        shapes=(tf.TensorShape((WINDOW_SIZE, WINDOW_SIZE, IMAGE_CHANNELS)), tf.TensorShape([NUM_CLASSES]))
        dataset = tf.data.Dataset.from_generator(
            gen, types, shapes
        )
        #dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size*8)
        return dataset

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        dataset_info = shuffle(dataset_info)
        while True:
            for xs, xe, ys, ye in data_generator.slice_images():
                for idx in range(len(dataset_info)):
                    #X_train_batch = dataset_info[start:end]
                    batch_labels = np.zeros((NUM_CLASSES))
                    image = data_generator.load_image(
                            dataset_info[idx]['path'], shape)   
                    if augument:
                        image = data_generator.augment(image)
                    #print(image)
                    image=image/255.
                    #print(image)
                    batch_labels[dataset_info[idx]['labels']] = 1
                    yield image[xs:xe, ys:ye, :], batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
        np.array(image_red_ch), 
        np.array(image_green_ch), 
        np.array(image_blue_ch)), -1)
        #image = cv2.resize(image, (shape[0], shape[1]))
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

    def slice_images():
        offset = int(IMAGE_SIZE%WINDOW_SIZE)
        for i in range(2):
            for j in range(2):
                x_start=i*offset
                x_end=x_start+WINDOW_SIZE

                y_start=j*offset
                y_end=y_start+WINDOW_SIZE
                
                yield x_start, x_end, y_start, y_end


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Reshape, Dense, Concatenate, GlobalMaxPooling2D
from keras.layers import BatchNormalization, Input, Conv2D, Lambda, Average
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Model

    
def create_model(n_out):
    input_shape=(WINDOW_SIZE,WINDOW_SIZE, IMAGE_CHANNELS)
    input_tensor = Input(shape=(WINDOW_SIZE, WINDOW_SIZE, IMAGE_CHANNELS))
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=input_shape
                             #input_shape=(WINDOW_SIZE, WINDOW_SIZE, IMAGE_CHANNELS)
                            )
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

# warm up model
model = create_model(n_out=NUM_CLASSES)
# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 10; batch_size = 16
checkpoint = ModelCheckpoint('../working/InceptionV3.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6)
callbacks_list = [checkpoint, early, reduceLROnPlat]

# split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

# create train and valid datagens
train_generator = data_generator.get_dataset(
    train_dataset_info[train_indexes], batch_size, (IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS), augument=True)
validation_generator = data_generator.get_dataset(
    train_dataset_info[valid_indexes], 32, (IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS), augument=False)

model.summary()
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
print(keras.__version__)
SVG(model_to_dot(model).create(prog='dot', format='svg'))



for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(1e-03),
    metrics=['acc'])
# model.summary()
train_images, train_labels = train_generator.make_one_shot_iterator().get_next()
val_images, val_labels = validation_generator.make_one_shot_iterator().get_next()
#model.fit(
#    x=train_images, y=train_labels,
#    steps_per_epoch=int(np.ceil(float(len(train_indexes)) / float(batch_size))*4),
#    validation_data=(val_images, val_labels),
#    validation_steps=int(np.ceil(float(len(valid_indexes)) / float(batch_size))*4),
#    epochs=2, 
#    verbose=1)
# train all layers
for layer in model.layers:
    layer.trainable = True
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])
#model.fit(
#    x=train_images, y=train_labels,
#    steps_per_epoch=int(np.ceil(float(len(train_indexes)) / float(batch_size))*4),
#    validation_data=(val_images, val_labels),
#    validation_steps=int(np.ceil(float(len(valid_indexes)) / float(batch_size))*4),
#    epochs=epochs,
#    verbose=1,
#    callbacks=callbacks_list)
# Create submit
submit = pd.read_csv(path_to_train+'/sample_submission.csv')
predicted = []
draw_predict = []
#model.load_weights('../working/InceptionV3.h5')
model.load_weights('../input/sliding-window-inception-v3-weights/InceptionV3.h5')

# This function is used to generate images for inference
def iter_image_slices(image):
    for x_start, x_end, y_start, y_end in data_generator.slice_images():
        yield image[x_start:x_end, y_start:y_end, :]

for name in tqdm(submit['Id']):
    path = os.path.join(path_to_train+'/test/', name)
    image = data_generator.load_image(path, (IMAGE_SIZE,IMAGE_SIZE,3))/255.
    classes=set()
    for s in iter_image_slices(image):
        score_predict = model.predict(s[np.newaxis])[0]
        draw_predict.append(score_predict)
        label_predict = np.arange(NUM_CLASSES)[score_predict>=0.2]
        classes.update(label_predict)
    str_predict_label = ' '.join(str(l) for l in classes)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('submit_InceptionV3.csv', index=False)

