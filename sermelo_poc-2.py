# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
import random
# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Conv2D, Dropout, MaxPooling2D, concatenate, UpSampling2D
import tensorflow.keras as keras

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Método creado por mi para poder mostrar más de dos historias
def multi_plot_compare(histories, names, title, field, legend_loc):
    """Compara losses de una lista de entrenamientos(histories) con nombres(names). Max 10"""
    legend = []
    for history, name, color in zip(histories, names, mcolors.TABLEAU_COLORS):
        plt.plot(history.history[field], color=color)
        plt.plot(history.history[f'val_{field}'], 'r--', color=color)
        legend.append('Train ' + name)
        legend.append('Val ' + name)
    plt.title(title)
    plt.ylabel(field)
    plt.xlabel('Epoch')
    plt.legend(legend,
               loc=legend_loc)
    plt.show()

def multi_plot_compare_losses(histories, names, title):
    multi_plot_compare(histories, names, f'{title} losses', 'loss', 'upper right')
    
def multi_plot_compare_accs(histories, names, title):
    multi_plot_compare(histories, names, f'{title} accuracies', 'acc', 'lower right')

def multi_plot_compare_histories(histories, names, title):
    multi_plot_compare_losses(histories, names, title)
    multi_plot_compare_accs(histories, names, title)
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
        return imageio.imread(f'{TRAINING_IMAGES_PATH}/{self.imageId}')

    def get_flat_grey_image(self):
        gray_image = rgb2gray(imageio.imread(f'{TRAINING_IMAGES_PATH}/{self.imageId}'))
        flat_gray_image = [item for sublist in gray_image for item in sublist]
        return flat_gray_image
    
    def get_masks(self):
        return self.masks

    def get_united_masks(self):
        united_mask = self.get_flat_united_mask()
        united_mask = np.reshape(united_mask, (768, 768)).T
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
unified_mask = np.zeros((img.shape[0], img.shape[1]))
for mask in masks:
   unified_mask += mask

print(unified_mask.shape)
print(img.shape)

mask2 = image.get_flat_united_mask()
unique, counts = np.unique(mask2, return_counts=True)
print(dict(zip(unique, counts)))
print(mask2.shape)
mask2 = np.reshape(mask2, (768, 768)).T

fig=plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 3, 1)
plt.imshow(img)
ax = fig.add_subplot(1, 3, 2)
plt.imshow(unified_mask)
ax = fig.add_subplot(1, 3, 3)
plt.imshow(mask2)
plt.show()

plt.show()      
def get_dataset_X_y(chosen_image, full_dataset):
    # TODO instanciate as a tensor from the begining
    X_data = []
    y_data = []
    c = 0
    for index, image_id in chosen_image.items():
        #print(image_id)
        img_obj = AirbusImage(image_id, full_dataset)
        X_data.append(tf.convert_to_tensor(img_obj.get_flat_grey_image(), dtype=tf.float32))
        mask = img_obj.get_flat_united_mask()
        y_data.append(tf.convert_to_tensor(mask, dtype=tf.bool))
        c += 1
        if c % 50 == 0:
            print(c)
    print('Converting to tensor')
    X_data = tf.convert_to_tensor(X_data)
    y_data = tf.convert_to_tensor(y_data)
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
#images_to_use, images_discarded = train_test_split(images_name, test_size=0.95)
images_to_use, images_discarded = train_test_split(images_name, test_size=0.9995)
print(f'After reducing the size {images_to_use.shape}')
train_df, test_df = train_test_split(images_to_use, test_size=0.3)
print(f'Training data shape: {train_df.shape}')
print(f'Test data shape: {test_df.shape}')

X_train, y_train = get_dataset_X_y(train_df, image_data)
X_test, y_test = get_dataset_X_y(test_df, image_data)
print(X_train.shape)
print(y_train.shape)

model = tf.keras.Sequential()
model.add(Input(shape=(768*768,)))
model.add(Dense(128))
model.add(ReLU(128))
model.add(BatchNormalization())#model.add(layers.Dense(16))
model.add(Dense(768*768))
model.add(ReLU(128))
#model.add(Conv2D(2, (1,) , padding='same'))
model.compile(optimizer='adam',
              #loss=keras.losses.binary_crossentropy,
              loss=keras.losses.categorical_crossentropy,
              #loss=keras.losses.sparse_categorical_crossentropy,
              metrics = ['acc'])

model.summary()
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=3,#epochs=30,
                    shuffle=True,
                    callbacks = [keras.callbacks.EarlyStopping(patience=3)],
                    #class_weight = {0: 0.1, 1: 0.9},
                    #class_weight = {0: 0.11, 1: 0.89},
                    validation_data=(X_test, y_test))
multi_plot_compare_histories([history],['Data'], 'Training')
test = model.predict(np.array([X_test[2]]))

test.shape
def plot_prediction(num_predictions):
    print(f'Predictions of {num_predictions} images')
    fig=plt.figure(figsize=(20, (num_predictions/6 * 25)))
    for i in range(0, num_predictions):
        image_num = random.randint(0,X_test.shape[0])
        print(f'Preparing prediction {i+1}. Choosen image number {image_num}')
        img = np.reshape(X_test[image_num], (768, 768))
        
        real_mask = np.reshape(y_test[image_num], (768, 768)).T
        
        prediction = model.predict(np.array([X_test[image_num]]))     
        predicted_mask = np.reshape(prediction[0], (768, 768)).T
        predicted_mask = predicted_mask > 0.5        
        
        ax = fig.add_subplot(num_predictions, 3, 1 + (i*3))
        ax.axis('off')
        plt.imshow(img)
        ax = fig.add_subplot(num_predictions, 3, 2 + (i*3))
        ax.axis('off')
        plt.imshow(real_mask)
        ax = fig.add_subplot(num_predictions, 3, 3 + (i*3))
        ax.axis('off')
        plt.imshow(predicted_mask)
    plt.show()

plot_prediction(10)
    
