import os
import cv2
import math
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
import tensorflow.keras as keras



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
print(os.listdir("../input"))
PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
SAMPLE = '../input/sample_submission.csv'
label_df = pd.read_csv('../input/train.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')
label_df.head()
label_df['Id'].describe()
# Display the most frequent ID (without counting new_whale)
label_df['Id'].value_counts()[1:16].plot(kind='bar')
n_classes = label_df['Id'].nunique()
img_shape = (224,224,3)
def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width


def pad_and_resize(image_path, dataset):
    img = cv2.imread(f'../input/{dataset}/{image_path}')
    pad_width = get_pad_width(img, max(img.shape))
    padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    resized = cv2.resize(padded, (224,224))
    
    return resized
data = pd.read_csv('../input/train.csv')

target_dummies = pd.get_dummies(label_df['Id'])
train_label = target_dummies.columns.values
y_train = target_dummies.values

train_dataset_info = []
for name, labels in zip(data['Image'], y_train):
    train_dataset_info.append({
        'path':os.path.join(TRAIN, name),
        'labels': labels})
train_dataset_info = np.array(train_dataset_info)
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Image'], data['Id'], test_size=0.1)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, n_labels, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, n_labels))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image

                batch_labels[i] = dataset_info[idx]['labels']
            yield batch_images, batch_labels
            

    def load_image(path, shape):
        img = cv2.imread(path)
        resized = cv2.resize(img,  (shape[0], shape[1]))
        resized = resized / 255
        return resized

            
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
# create visualization datagen
vis_datagen = data_generator.create_train(
    train_dataset_info, 5, img_shape, n_classes, augument=False)

images, labels = next(vis_datagen)
fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
batchsize = 64
# create train datagen
train_datagen = data_generator.create_train(
    train_dataset_info[train_ids.index], batchsize, img_shape, n_classes, augument=True)

validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, img_shape, n_classes, augument=False)
def gen_graph(history, title):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation',], loc='upper left')
    plt.show()
    plt.plot(history.history['categorical_crossentropy'])
    plt.plot(history.history['val_categorical_crossentropy'])
    plt.title('Loss ' + title)
    plt.ylabel('MLogLoss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
STEPS = 512
epochs = 15
def create_model(input_shape, n_out):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None, classes=n_out)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dense(n_out)(x)
    logits = Activation('softmax')(x)
    
    for layer in base_model.layers:
        layer.trainable = True
        
    return Model(inputs=base_model.input, outputs=logits)


model = create_model(input_shape=img_shape, n_out=n_classes)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy])
model.summary()
callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1)
]

hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=epochs, verbose=1,
    validation_data=next(validation_generator),
    callbacks = callbacks)
#plot
gen_graph(hist, 
              "Mobile Net, lr 1e-4")