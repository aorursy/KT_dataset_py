# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import argmax
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from skimage.measure import label
from skimage.measure import regionprops
import imageio
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
import random
# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.activations import softmax, sigmoid

import tensorflow.keras as keras
import keras.backend as K
import matplotlib.pyplot as plt
DATA_PATH = f'/kaggle/input/airbus-ship-detection/'
for x in os.listdir(DATA_PATH):
    print(x)
    
TRAINING_CSV = f'{DATA_PATH}/train_ship_segmentations_v2.csv'
TRAINING_IMAGES_PATH = f'{DATA_PATH}/train_v2/'
TEST_IMAGES_PATH = f'{DATA_PATH}/test_v2/'


OUTPUT_PATH = f'/kaggle/output'

Path(OUTPUT_PATH).mkdir(exist_ok=True)
for x in os.listdir(OUTPUT_PATH):
    print(x)
class AirbusImage(object):
    images_path = TRAINING_IMAGES_PATH
    def __init__(self, imageId, data):
        self.imageId = imageId
        self.image = None
        image_data = data.loc[data['ImageId'] == self.imageId, 'EncodedPixels'].tolist()
        self.flat_masks, self.masks, self.labels, self.boxes = self.__create_boats_data(image_data)

    def __rle_decode_flatten(self, mask_rle):
        shape=(768, 768)
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img

    def __rle_decode(self, mask_rle):
        shape=(768, 768)
        flat_mask = self.__rle_decode_flatten(mask_rle)
        return flat_mask.reshape(shape).T

    def __create_boats_data(self, data):
        masks = []
        flat_masks = []
        labels = []
        boxes = []
        if data != [-1]:
            for encoded_mask in data:
                img_mask = self.__rle_decode(encoded_mask)
                flat_masks.append(self.__rle_decode_flatten(encoded_mask))
                masks.append(img_mask)
                img_label = label(img_mask)
                labels.append(img_label)
                img_box = regionprops(img_label)[0].bbox
                boxes.append(img_box)
        return flat_masks, masks, labels, boxes

    def get_image(self):        
        image = rgb2gray(imageio.imread(f'{TRAINING_IMAGES_PATH}/{self.imageId}'))
        #image_resized = resize(image, (512, 512), anti_aliasing=True)
        #image_resized = np.reshape(image_resized, (512, 512, 1))
        image_resized = resize(image, (256, 256), anti_aliasing=True)
        image_resized = np.reshape(image_resized, (256, 256, 1))
        return image_resized

    def get_flat_grey_image(self):
        gray_image = rgb2gray(imageio.imread(f'{TRAINING_IMAGES_PATH}/{self.imageId}'))
        flat_gray_image = [item for sublist in gray_image for item in sublist]
        return flat_gray_image
    
    def get_masks(self):
        return self.masks

    def get_united_masks(self):
        united_mask = self.get_flat_united_mask()
        united_mask = np.reshape(united_mask, (768, 768)).T
        united_mask = resize(united_mask, (256, 256), anti_aliasing=True)
        return united_mask
    
    def get_flat_masks(self):
        return self.flat_masks

    def get_flat_united_mask(self):
        unite_mask = np.zeros(768 * 768, dtype=np.uint8)
        for mask in self.flat_masks:
            unite_mask += mask
        return unite_mask
    
    def get_boxes(self):
        return self.boxes

    def get_labels(self):
        return self.labels
    
    def get_height(self):
        return 768
    
    def get_width(self):
        return 768
    
    def get_encoded_jpg(self):
        with tf.gfile.GFile(f'{TRAINING_IMAGES_PATH}/{self.imageId}', 'rb') as fid:
            encoded_jpg = fid.read()
        return encoded_jpg
from skimage.draw import rectangle_perimeter

data = pd.read_csv(TRAINING_CSV).fillna(-1)
image_name = 'c8b051d24.jpg'
image = AirbusImage(image_name, data)

img = image.get_image()
masks = image.get_masks()
#print(masked_image.shape)
unified_mask = image.get_united_masks()

fig=plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 3, 1)
img = np.reshape(img, (256, 256))
plt.imshow(img)
ax = fig.add_subplot(1, 3, 2)
plt.imshow(unified_mask)
ax = fig.add_subplot(1, 3, 3)
plt.imshow(unified_mask)
plt.show()
def get_dataset_X_y(chosen_image, full_dataset):
    X_data = []
    y_data = []
    c = 0
    for index, image_id in chosen_image.items():
        #print(image_id)
        img_obj = AirbusImage(image_id, full_dataset)
        X_data.append(tf.convert_to_tensor(img_obj.get_image()/255, dtype=tf.float32))
        mask = img_obj.get_united_masks()
        y_data.append(tf.convert_to_tensor(mask, dtype=tf.bool))
        c += 1
        if c % 50 == 0:
            print(c)
    print('Converting to tensor')
    X_data = tf.convert_to_tensor(X_data)
    y_data = tf.convert_to_tensor(y_data)#, dtype=tf.float32)
    y_data = keras.utils.to_categorical(y_data, 2)
    print('End converting to tensor')
    return X_data, y_data


image_data = pd.read_csv(TRAINING_CSV).dropna()
image_data = image_data.reset_index(drop=True)

images_name = image_data['ImageId']
images_name = images_name.drop_duplicates()
images_name = images_name.sample(frac=1) # shuffle
print(f'Original images with data {images_name.shape}')
#print(images_name.head())
#images_to_use, images_discarded = train_test_split(images_name, test_size=0.995)
#images_to_use, images_discarded = train_test_split(images_name, test_size=0.97)
images_to_use, images_discarded = train_test_split(images_name, test_size=0.95)
print(f'After reducing the size {images_to_use.shape}')
train_df, test_df = train_test_split(images_to_use, test_size=0.3)
print(f'Training data shape: {train_df.shape}')
print(f'Test data shape: {test_df.shape}')

X_train, y_train = get_dataset_X_y(train_df, image_data)
X_test, y_test = get_dataset_X_y(test_df, image_data)

print(f'Shape of element 0 of X {X_train[0].shape}')
print(f'Shape of element 0 of y {y_train[0].shape}')
#print(X_train)
print(y_train)
def customLoss(yTrue, yPred):
    weights = {0: 1, 1:155}
    diff = yTrue - yPred

    onesCondition = K.cast(K.equal(yTrue, 0.), tf.float32) * weights[0]
    zeroCondition = K.cast(K.not_equal(yTrue, 0.), tf.float32) * weights[1]
    weighted_tensor = onesCondition + zeroCondition
    
    diff = diff * weighted_tensor
    loss = K.mean(K.abs(diff))
    #print(loss)
    return loss

sol = customLoss(tf.convert_to_tensor([0., 0., 0., 0.], dtype=tf.float32), tf.convert_to_tensor([0.,0.,0.,1.], dtype=tf.float32))
sol
#print(sol)
#print(sol.gradient())
#tf.gradients(sol)

def oneHotcustomLoss(yTrue, yPred):
    #weights = {0: 1, 1:155}
    weighted = (K.argmax(yTrue) * 500) + 1
    #print(weighted)
    return tf.losses.binary_crossentropy(yTrue, yPred) * K.cast(weighted, tf.float32)

#K.argmax(y_train[4])
true = tf.convert_to_tensor([[[1, 0],[0, 1],[1, 0]]], tf.float32)
#pred = tf.convert_to_tensor([[[0, 1],[1, 0],[0, 1]]], tf.float32)
pred = tf.convert_to_tensor([[[0.5, 0.5],[1, 0],[0, 1]]], tf.float32)

oneHotcustomLoss(true, pred)

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)


import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
n_classes = 2

img_input = Input(shape=(256, 256, 1))

# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)

conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)

# Dencoder
conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
up1 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=-1)

conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up1)
conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
up2 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=-1)

conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up2)
conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
up3 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=-1)

conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)
conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
up4 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=-1)

# Output
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
conv9 = Conv2D(n_classes, (3, 3), activation='relu', padding='same')(conv9)


out = softmax(conv9)

model = keras.Model(inputs=img_input, outputs=out, name="AirbusCNN")

model.summary()

model.compile(optimizer='adam',
              #loss=WeightedBinaryCrossEntropy(pos_weight=0.5, weight = 2, from_logits=True),
              loss=oneHotcustomLoss,
              #loss=dice_p_bce,
              #loss=IoU,
              metrics = ['acc'])

history = model.fit(X_train, y_train,
              batch_size=32,
              epochs=15,
              shuffle=True,
              callbacks = [keras.callbacks.EarlyStopping(patience=5)],
              validation_data=(X_test, y_test))
def plot_prediction(num_predictions):
    print(f'Predictions of {num_predictions} images')
    fig=plt.figure(figsize=(20, (num_predictions/6 * 25)))
    for i in range(0, num_predictions):
        image_num = random.randint(0,X_test.shape[0])
        print(f'Preparing prediction {i+1}. Choosen image number {image_num}')
        img = X_test[image_num] #np.reshape(X_test[image_num], (768, 768, 3))
        img = np.reshape(img, (256, 256))
        
        real_mask = argmax(y_test[image_num], axis=2)
        #real_mask = y_test[image_num]
        print('Going to do a prediction')
        prediction = model.predict(np.array([X_test[image_num]]))
        print('Prediction done')
        predicted_mask = argmax(prediction[0], axis=2)
        #predicted_mask = np.reshape(prediction[0], (256, 256))

        
        ax = fig.add_subplot(num_predictions, 3, 1 + (i*3))
        ax.axis('off')
        plt.imshow(img)
        ax = fig.add_subplot(num_predictions, 3, 2 + (i*3))
        ax.axis('off')
        print(real_mask.shape)
        plt.imshow(real_mask)
        ax = fig.add_subplot(num_predictions, 3, 3 + (i*3))
        ax.axis('off')
        plt.imshow(predicted_mask)
    plt.show()

plot_prediction(16)


#y_test[1]
#argmax(y_test[1], axis=2)