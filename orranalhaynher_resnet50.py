import cv2

import numpy as np 

import pandas as pd

from keras.backend import set_image_data_format

from tensorflow.keras.preprocessing import image

from skimage.io import imread_collection, imread

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
def get_order_filenames(file_path):

    df = pd.read_csv(file_path)

    return df["image"]
file_path = '../input/specialist-segmentation/reading_order.csv'

image_names = get_order_filenames(file_path)
path = '../input/kmeanssegmentation/'

gray_images = [imread(path+str(name)+'.bmp') for name in image_names]

print('The database has {} segmented images'.format(len(gray_images)))
images = np.zeros((len(gray_images), gray_images[0].shape[0], gray_images[0].shape[1], 3))



for i, im in enumerate(gray_images):

    for j in range(3):

        images[i, :, :, j] = im
model = ResNet50(weights='imagenet', include_top=False, input_shape=(217, 323, 3), pooling='avg')
x = preprocess_input(images)



features = model.predict(x)
features[0].shape
np.save('./resnet50Kmeans.npy', features)