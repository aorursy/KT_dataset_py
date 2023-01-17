!pip install imutils

!pip install efficientnet

!pip install iterative-stratification

!pip install albumentations > /dev/null

!pip install image-classifiers==1.0.0b1
import os

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

import os

import random

import imutils

import matplotlib.pyplot as plt

import collections

from tqdm import tqdm_notebook as tqdm

from datetime import datetime



from math import ceil, floor, log

import cv2

from scipy import ndimage

import sys



import tensorflow as tf

from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Add, GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,concatenate

from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, LearningRateScheduler

from tensorflow.keras.optimizers import Adam,RMSprop

from tensorflow.keras.models import Model,load_model



from sklearn.model_selection import ShuffleSplit

from albumentations import (

    Compose, HorizontalFlip, CLAHE, HueSaturationValue,

    RandomBrightness, RandomContrast, RandomGamma,OneOf,

    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue, VerticalFlip,

    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,Normalize,

    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop,RandomCrop,RandomResizedCrop,RandomRotate90,Transpose

)



import efficientnet.tfkeras as efn

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

from classification_models.tfkeras import Classifiers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input d
df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')
train_images_path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'

test_images_path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'

train_files = os.listdir(train_images_path)
hem_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')

hem_df['sub_type'] = hem_df['ID'].str.split('_',expand = True)[2]

hem_df['image'] = 'ID_' + hem_df['ID'].str.split('_',expand = True)[1] + '.dcm'

hem_df = hem_df.pivot_table(index = 'image', columns = 'sub_type')

hem_df
def correct_dcm(dcm):

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000

    

def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):

    img = window_image(img, window_center, window_width)

    ue = np.log((U / eps) - 1.0)

    W = (2 / window_width) * ue

    b = ((-2 * window_center) / window_width) * ue

    z = W * img + b

    img = U / (1 + np.power(np.e, -1.0 * z))

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img





def window_image(dcm, window_center, window_width, desired_size):

    

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):

        correct_dcm(dcm)

    

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img = cv2.resize(img, desired_size[:2], interpolation = cv2.INTER_AREA)  # resize image

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)



    return img



def bsb_window(dcm, desired_size):

    brain_img = window_image(dcm, 40, 80, desired_size)

    subdural_img = window_image(dcm, 80, 200, desired_size)

    soft_img = window_image(dcm, 40, 380, desired_size)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img
def read_img_val(path, desired_size):

    dcm = pydicom.dcmread(path)

    try:

        img = bsb_window(dcm, desired_size)

    except:

        img = np.zeros(desired_size)

        

    return img
class GradCAM:

	def __init__(self, model, classIdx, layerName=None):



		self.model = model

		self.classIdx = classIdx

		self.layerName = layerName



		

		if self.layerName is None:

			self.layerName = self.find_target_layer()



	def find_target_layer(self):



		for layer in reversed(self.model.layers):

			if len(layer.output_shape) == 4:

				return layer.name



	

		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")



	def compute_heatmap(self, image, eps=1e-8):



		gradModel = Model(

			inputs=[self.model.inputs],

			outputs=[self.model.get_layer(self.layerName).output, 

                     

				self.model.output])



		with tf.GradientTape() as tape:

		

			inputs = tf.cast(image, tf.float32)

			(convOutputs, predictions) = gradModel(inputs)

			loss = predictions[:, self.classIdx]



		grads = tape.gradient(loss, convOutputs)



		castConvOutputs = tf.cast(convOutputs > 0, "float32")

		castGrads = tf.cast(grads > 0, "float32")

		guidedGrads = castConvOutputs * castGrads * grads





		convOutputs = convOutputs[0]

		guidedGrads = guidedGrads[0]



	

		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))

		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)





		(w, h) = (image.shape[2], image.shape[1])

		heatmap = cv2.resize(cam.numpy(), (w, h))



	

		numer = heatmap - np.min(heatmap)

		denom = (heatmap.max() - heatmap.min()) + eps

		heatmap = numer / denom

		heatmap = (heatmap * 255).astype("uint8")



		return heatmap



	def overlay_heatmap(self, heatmap, image, alpha=0.5,

		colormap=cv2.COLORMAP_JET):



		heatmap = cv2.applyColorMap(heatmap, colormap)

		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)



	

		return (heatmap, output)
def create_model():

    

    base_model =  efn.EfficientNetB4(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (256,256,3))

    x = base_model.output

    x = Dropout(0.15)(x)

    y_pred = Dense(6, activation = 'sigmoid')(x)



    return Model(inputs = base_model.input, outputs = y_pred)

model = create_model()

model.load_weights('../input/efficientnetb4-hemorrhage/efficientnetb4_model.h5')
# random_integer = random.randint(0,len(hem_df) - 1)

# random_img = img_ids[-3]

# image = read_img_val(train_images_path + random_img,(256, 256))

# label = hem_df.iloc[-3].Label.values
hem_df[hem_df.Label.subarachnoid.values == 1].sample(20)
subdural_img = 'ID_29c9c2ee8.dcm'	

intraventricular_img = 'ID_004780f8e.dcm'

intraparenchymal_img = 'ID_000d69988.dcm'

subarachnoid_img = 'ID_00058bb06.dcm'

epidural_img = 'ID_00f1e66e1.dcm'

epidural_img2 = 'ID_ff0afaa64.dcm'

subdural_img2 = 'ID_47f130bfa.dcm'	

subarachnoid_img2 = 'ID_526b45786.dcm'



type_list = [subdural_img, intraventricular_img, intraparenchymal_img, subarachnoid_img, epidural_img2]
img_id = subarachnoid_img2

img = hem_df.loc[hem_df.index == img_id].index[0]

image = read_img_val(train_images_path + img,(256, 256))

label = hem_df.loc[hem_df.index == img_id].Label.values
plt.imshow(image)
class_label = ['any', 'EPH', 'IPH','IVH', 'SAH', 'SDH']
preds = model.predict(image[np.newaxis,...])

i = np.argmax(preds[0])



# decode the ImageNet predictions to obtain the human-readable label



# # initialize our gradient class activation map and build the heatmap

cam = GradCAM(model, i)

heatmap = cam.compute_heatmap(image[np.newaxis,...])



img_copy = np.copy(image)

img_copy -= img_copy.min((0,1))

img_copy = (255*img_copy).astype(np.uint8)

# resize the resulting heatmap to the original input image dimensions

# and then overlay heatmap on top of the image

heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

(heatmap, output) = cam.overlay_heatmap(heatmap, img_copy, alpha=0.5)

print("Hemorrhage type: {}".format(class_label))

print("actual label: {}".format(label))

print("predicted label: {}".format(preds))
plt.imshow(output)
img_label_list = []

for img_type in type_list:

    img_id = img_type

    img = hem_df.loc[hem_df.index == img_id].index[0]

    image = read_img_val(train_images_path + img,(256, 256))

    label = hem_df.loc[hem_df.index == img_id].Label.values

    img_label_list.append((image, label))


def plot_map(img_label_list):

    fig, axes = plt.subplots(5, 2, figsize=(15, 15))

    fig.suptitle('Grad-CAM\nPredicted / Actual / Probability',fontsize=20)

    

    for i, img_label in enumerate(img_label_list):

        img, label = img_label

        preds = model.predict(img[np.newaxis,...])

        axes[i,0].imshow(img, cmap = 'bone')

        axes[i,0].set_xticks([])

        axes[i,0].set_yticks([])

        axes[i,0].set_title(f'{class_label[np.argmax(preds[:, 1:]) + 1]} / {class_label[np.argmax(label[:, 1:]) + 1]} / {np.max(preds[:, 1:]):.4f}')

        heatmap = cam.compute_heatmap(img[np.newaxis,...])

        img_copy = np.copy(img)

        img_copy -= img_copy.min((0,1))

        img_copy = (255*img_copy).astype(np.uint8)

        # resize the resulting heatmap to the original input image dimensions

        # and then overlay heatmap on top of the image

        heatmap = cv2.resize(heatmap, (img_copy.shape[1], img_copy.shape[0]))

        (heatmap, output) = cam.overlay_heatmap(heatmap, img_copy, alpha=0.5)

        axes[i,1].imshow(output)

        axes[i,1].set_xticks([])

        axes[i,1].set_yticks([])

        axes[i,1].set_title("heatmap showing hemorrhage location")

    plt.subplots_adjust(wspace=1, hspace=0.2)

    plt.savefig('hemorrhageGradCAM.png')

plot_map(img_label_list)