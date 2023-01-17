# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

stage_2_detailed_class_info = pd.read_csv("../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv")

stage_2_sample_submission = pd.read_csv("../input/rsna-pneumonia-detection-challenge/stage_2_sample_submission.csv")

stage_2_train_labels = pd.read_csv("../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
import os 

import sys

import random

import math

import numpy as np

import cv2

import matplotlib.pyplot as plt

import json

import pydicom

import pandas as pd 

import glob
DATA_DIR = "../input/rsna-pneumonia-detection-challenge/"

ROOT_DIR = '/kaggle/working'
train_dicom_dir = os.path.join(DATA_DIR, 'stage_2_train_images')

test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')
df_1 = pd.merge(stage_2_detailed_class_info,stage_2_train_labels,on = 'patientId')

print(df_1.head(10))

print(df_1.shape)
g_Train = glob.glob(train_dicom_dir + '/*.dcm')

g_Test = glob.glob(test_dicom_dir + '/*.dcm')
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g_Train))

print ('\n'.join(g_Train[:5]))

print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g_Test))

print ('\n'.join(g_Test[:5]))
%%time



PathDicom = train_dicom_dir

lstFilesDCM = []  # create an empty list

FileName = []

annotations = {}

for dirName, subdirList, fileList in os.walk(PathDicom):

    for filename in fileList:

        if ".dcm" in filename.lower():  # check whether the file's DICOM

            lstFilesDCM.append(os.path.join(dirName,filename))

            FileName.append(filename.strip('.dcm'))
df_2 = pd.DataFrame({'patientId':FileName,'path':lstFilesDCM})

print(df_2.head(10))

print(df_2.shape)
df_3 = pd.merge(df_1,df_2,on = 'patientId')

print(df_3.head(10))

print(df_3.shape)
image_annotations = {fp: [] for fp in lstFilesDCM}

len(image_annotations)
anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train_labels.csv'))

print(anns.head(10))

print(anns.shape)

for index, row in anns.iterrows(): 

    fp = os.path.join(PathDicom, row['patientId']+'.dcm')

    image_annotations[fp].append(row)
image_annotations
len(image_annotations)
#pd.DataFrame.from_dict(data)

List_1 = [(k, v) for k, v in image_annotations.items()]

print(List_1[:10])

print(len(List_1))
df_4 = pd.DataFrame(List_1)

print(df_4.head(10))

print(df_4.shape)
df_4.rename(columns={0: "path_1", 1:"annotation"}, inplace = True)

print(df_4.head(10))

print(df_4.shape)
df_4['annotation'][0][0][0]
patientId_df = []

for index, row in df_4.iterrows():

    patientId_df.append(df_4['annotation'][index][0][0])

len(patientId_df)
df_4['patientId'] = patientId_df

print(df_4.head(10))

print(df_4.shape)
df_5 = pd.merge(df_3,df_4,on = 'patientId')

print(df_5.head(10))

print(df_5.shape)
IMAGE_WIDTH = 1024

IMAGE_HEIGHT = 1024
df_5['original_height'] = IMAGE_HEIGHT
df_5['original_width'] = IMAGE_WIDTH
df_5.head(10)
df_5.drop(['path_1'],axis=1,inplace = True)

df_5.head(10)
df_5.head(10)
df_5.to_csv('/kaggle/working/Data_Prep.csv')

np.save('Data_Prep.npy',df_5)
import os 

import sys

import random

import math

import numpy as np

import cv2

import matplotlib.pyplot as plt

import json

import pydicom

import pandas as pd 

import glob

os.getcwd()
Dataset = np.load('/kaggle/working/Data_Prep.npy',allow_pickle = True)
Dataset.astype
Dataset_DB =  pd.DataFrame(data=Dataset)
Dataset_DB.shape
Dataset_DB.rename(columns={0: "patientId", 1:"class",2:'x',3:'y',4:'width',5:'height',6:'Target',7:'path',8:'annotation',9:'original_height',10:'original_width'}, inplace = True)
Dataset_DB.head(2)
from random import randint

#Image_ID = 100

Image_ID = randint(0,29280)

fb = Dataset_DB['path'][Image_ID]

image_info = pydicom.read_file(fb)

image = image_info.pixel_array

plt.imshow(image,cmap='gray')

print(Dataset_DB['patientId'][Image_ID])

print(image.shape)
if len(image.shape) != 3 or image.shape[2] != 3:

            image = np.stack((image,) * 3, -1)

image.shape
annotation = Dataset_DB['annotation'][Image_ID]

count = len(annotation)

print(count)

if count == 0:

    mask = np.zeros((Dataset_DB['original_height'][Image_ID], Dataset_DB['original_width'][Image_ID], 1), dtype=np.uint8)

else:

    mask = np.zeros((Dataset_DB['original_height'][Image_ID], Dataset_DB['original_width'][Image_ID], count), dtype=np.uint8)

   

    for i, a in enumerate(annotation):

        if a['Target'] == 1:

            x = int(a['x'])

            y = int(a['y'])

            w = int(a['width'])

            h = int(a['height'])

            mask_instance = mask[:, :, i].copy()

            cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)

            mask[:, :, i] = mask_instance
masked = np.zeros(image.shape[:2])

for i in range(mask.shape[2]):

    masked += image[:, :, 0] * mask[:, :, i]

plt.imshow(masked, cmap='RdGy_r')

plt.axis('off')
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)

plt.imshow(image)

plt.axis('off')



plt.subplot(1, 2, 2)

masked = np.zeros(image.shape[:2])

for i in range(mask.shape[2]):

    masked += image[:, :, 0] * mask[:, :, i]

plt.imshow(masked, cmap='gray')

plt.axis('off')



print(Image_ID)

print(fb)

print(Dataset_DB['patientId'][Image_ID])
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import MobileNet, preprocess_input

from keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape

from keras.models import Model, Sequential

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping , ModelCheckpoint,ReduceLROnPlateau,Callback

from keras.utils import Sequence
ALPHA = 1.0

IMAGE_WIDTH = 224

IMAGE_HEIGHT = 224

EPOCHS = 5

BATCH_SIZE = 8

PATIENCE = 50

MULTI_PROCESSING = False

THREADS = 1


fb
Dataset_DB.head(10)
import cv2

from sklearn.model_selection import train_test_split  

from skimage.transform import resize

tarining_pd = Dataset_DB.head(2000)

x_train, x_test, y_train, y_test = train_test_split(tarining_pd, tarining_pd.Target, test_size=0.80, random_state=42)



masks = np.zeros((int(x_train.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH))

X_train = np.zeros((int(x_train.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH, 3))



#fb1 = x_train['path'].iloc[1]

#image_info1 = pydicom.read_file(fb)

#img1 = image_info.pixel_array

#img1 = cv2.resize(img1, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)

#x = preprocess_input(np.array(img1, dtype=np.float32))

#print(img1)

#print('------------------------------')

#print(x)

#print('------------------------------')

#print(x_train)

#print('------------------------------')

#X_train[1] = x

#print(X_train[1])

for index in range(x_train.shape[0]):

    fb = x_train['path'].iloc[index]

    image_info = pydicom.read_file(fb)

    #image_info = resize(image_info, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    img = image_info.pixel_array

   # print(img.shape)

    img = cv2.resize(img, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)

    #img = img[:,:,np.newaxis]

    img = np.stack((img,)*3, axis=-1)

    X_train[index] = preprocess_input(np.array(img, dtype=np.float32))

    #print(X_train[index])

    for i in X_train[index]:

       # print(i)

        x1 = int(i[0][0] * IMAGE_WIDTH)

        x2 = int(i[1][1] * IMAGE_WIDTH)

        y1 = int(i[0][0] * IMAGE_HEIGHT)

        y2 = int(i[1][1] * IMAGE_HEIGHT)

        masks[index][y1:y2, x1:x2] = 1
X_train[1]
def create_model(trainable=True):

    # model = #### Add your code here ####

    model = MobileNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), 

                      include_top=False, alpha=1.0, weights='imagenet')

    for layer in model.layers:

        layer.trainable = trainable



    # Add all the UNET layers here

    #### Add your code here ####



    # getting the layers from mobilenet network

    conv_pw_13_relu = model.get_layer("conv_pw_13_relu").output

    conv_pw_12_relu = model.get_layer("conv_pw_12_relu").output

    conv_pw_11_relu = model.get_layer("conv_pw_11_relu").output

    conv_pw_10_relu = model.get_layer("conv_pw_10_relu").output

    conv_pw_9_relu = model.get_layer("conv_pw_9_relu").output

    conv_pw_8_relu = model.get_layer("conv_pw_8_relu").output

    conv_pw_7_relu = model.get_layer("conv_pw_7_relu").output

    conv_pw_6_relu = model.get_layer("conv_pw_6_relu").output

    conv_pw_5_relu = model.get_layer("conv_pw_5_relu").output

    conv_pw_4_relu = model.get_layer("conv_pw_4_relu").output

    conv_pw_3_relu = model.get_layer("conv_pw_3_relu").output

    conv_pw_2_relu = model.get_layer("conv_pw_2_relu").output

    conv_pw_1_relu = model.get_layer("conv_pw_1_relu").output

    input_1 = model.layers[0].output



    

    # Adding Unet layers

    # Each set will have 1 upsampling, then concat with the mobilenet layers having same shape

    # followed by 2 conved layers with extra parameters



    up2 = UpSampling2D()(conv_pw_13_relu)

    concat1 = Concatenate()([up2, conv_pw_11_relu])

    new_conv15 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat1)

    new_conv15 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(new_conv15)



    up3 = UpSampling2D()(concat1)

    concat2 = Concatenate()([up3, conv_pw_5_relu])

    new_conv16 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat2)

    new_conv16 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(new_conv16)



    up4 = UpSampling2D()(concat2)

    concat3 = Concatenate()([up4, conv_pw_3_relu])

    new_conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat3)

    new_conv17 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(new_conv17)



    up5 = UpSampling2D()(concat3)

    concat4 = Concatenate()([up5, conv_pw_1_relu])

    new_conv17 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat4)

    new_conv17 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(new_conv17)



    up6 = UpSampling2D()(concat4)

    concat5 = Concatenate()([up6, input_1])



    outputs = Conv2D(1, kernel_size=1, activation="sigmoid")(concat5)

    outputs = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH))(outputs)



    # #### Add your code here ####

    return Model(inputs=model.input, outputs=outputs)    
model = create_model()



# Print summary

model.summary()
import tensorflow as tf

def dice_coefficient(y_true, y_pred):

    numerator = 2 * tf.reduce_sum(y_true * y_pred)

    denominator = tf.reduce_sum(y_true + y_pred)



    return numerator / (denominator + tf.keras.backend.epsilon())
from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.backend import log, epsilon

def loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())
model.compile(optimizer='Adam', loss=loss, metrics=[dice_coefficient])
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,

                             save_weights_only=True, mode="min", period=1)

stop = EarlyStopping(monitor="loss", patience=2, mode="min")

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
print(X_train.shape)

print(masks.shape)

model.fit(X_train,masks, epochs=30,batch_size = 1, verbose=1, callbacks=[checkpoint, reduce_lr, stop])

#from google.colab.patches import cv2_imshow

n = 10

THRESHOLD = 0.1

EPSILON = 0.02

HEIGHT_CELLS = 28

WIDTH_CELLS = 28



CELL_WIDTH = IMAGE_WIDTH / WIDTH_CELLS

CELL_HEIGHT = IMAGE_HEIGHT / HEIGHT_CELLS



sample_image =X_train[1]

print(sample_image)

feat_scaled = preprocess_input(np.array(sample_image, dtype=np.float32))

region = model.predict(x=np.array([feat_scaled]))[0]

#np.zeros((int(x_train.shape[0]), IMAGE_HEIGHT, IMAGE_WIDTH, 3))

output = np.zeros(sample_image.shape[:2], dtype=np.uint8)

#output = np.zeros(int(X_train.shape[0]), dtype=np.uint8)

for i in range(region.shape[1]):

    for j in range(region.shape[0]):

        if region[i][j] > THRESHOLD:

            x = int(CELL_WIDTH * j * sample_image.shape[1] / IMAGE_WIDTH)

            y = int(CELL_HEIGHT * i * sample_image.shape[0] / IMAGE_HEIGHT)

            x2 = int(CELL_WIDTH * (j + 1) * sample_image.shape[1] / IMAGE_WIDTH)

            y2 = int(CELL_HEIGHT * (i + 1) * sample_image.shape[0] / IMAGE_HEIGHT)



            output[y:y2,x:x2] = 1







X0 = ((sample_image[0]) * IMAGE_WIDTH / IMAGE_HEIGHT)



contours,hierachy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



#print(contours) 

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, EPSILON * cv2.arcLength(cnt, True), True)

    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 1)



plt.imshow(sample_image)

plt.show()

#cv2.waitKey(0)

#cv2.imshow("image", sample_image)

#cv2.waitKey(0)

#cv2.destroyAllWindows()
feat_scaled = preprocess_input(np.array(sample_image, dtype=np.float32))



pred_mask = cv2.resize(1.0*(model.predict(x=np.array([feat_scaled]))[0] > 0.5), (IMAGE_WIDTH,IMAGE_HEIGHT))



image2 = sample_image

image2[:,:,0] = pred_mask*sample_image[:,:,0]

image2[:,:,1] = pred_mask*sample_image[:,:,1]

image2[:,:,2] = pred_mask*sample_image[:,:,2]



out_image = image2

plt.imshow(out_image)

plt.show()