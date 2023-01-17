# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import openslide
import os
import cv2
import PIL
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16,preprocess_input
import plotly.graph_objs as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import skimage.io
from sklearn.model_selection import KFold
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.python.keras import backend as K
sess = K.get_session()
train=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
test=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv')
train.head()
test.head()
labels=[]
data=[]
data_dir='/kaggle/input/prostate-cancer-grade-assessment/train_images/'
for i in range(train.shape[0]):
    data.append(data_dir + train['image_id'].iloc[i]+'.jpg')
    labels.append(train['isup_grade'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['isup_grade']=labels
df
from keras.preprocessing import image
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['isup_grade'], test_size=0.2, random_state=1234)
train=pd.DataFrame(X_train)
train.columns=['images']
train['isup_grade']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['isup_grade']=y_val

train['isup_grade']=train['isup_grade'].astype(str)
validation['isup_grade']=validation['isup_grade'].astype(str)

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True)

val_datagen=train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validate_filenames=False)

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validate_filenames=False)
train=pd.DataFrame(X_train)
train.columns=['images']
train['target']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['target']=y_val

train['target']=train['target'].astype(str)
validation['target']=validation['target'].astype(str)

from keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()
def vgg16_model( num_classes=None):

    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x=Flatten()(model.output)
    x=Dropout(0.5)(x)
    output=Dense(num_classes,activation='softmax')(x)
    model=Model(model.input,output)
    return model

vgg_conv=vgg16_model(6)
vgg_conv.summary()
def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)
opt = SGD(0.001,momentum=0.9,decay=1e-4)
vgg_conv.compile(loss='categorical_crossentropy',optimizer=opt,metrics=[kappa_score])
df.shape
nb_epochs = 5
batch_size=32
nb_train_steps = train.shape[0]//batch_size
nb_val_steps=validation.shape[0]//batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
train['images'][9645]
'''vgg_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_val_steps)'''
'''
train_image = []
for i in range(train.shape[0]):
    img = image.load_img('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'].iloc[i]+'.jpg', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
'''
labels=[]
data=[]
data_dir='/kaggle/input/prostate-cancer-grade-assessment/train_images'
for i in range(train.shape[0]):
    data.append(data_dir + train['image_id'].iloc[i]+'.jpg')
    labels.append(train['isup_grade'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['target']=labels

import glob
import cv2
import numpy as np
pic_num=1
IMG_DIR='/kaggle/input/prostate-cancer-grade-assessment/train_images'
def read_images(train):
    for img in glob.glob(train+"/*.jpg"):
        image = cv2.imread(img)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_image = cv2.resize(image/255.0  , (32 , 32))
        #cv2.imwrite("small/"+str(pic_num)+'.jpg',resized_img)

        yield train_image

train_image =  np.array(list(read_images(IMG_DIR)))
train_image
train=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
test=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv')
train
import os, shutil
from functools import reduce

import pandas as pd
import skimage.io
from skimage.io import imshow,show
from skimage.transform import resize
from skimage.util import montage
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import gc



def compute_statistics(image,colors_summed):
    """
    Args:
        image                  numpy.array   multi-dimensional array of the form WxHxC
    
    Returns:
        ratio_white_pixels     float         ratio of white pixels over total pixels in the image 
    """
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height
    
    num_white_pixels = 0
    
    #summed_matrix = np.sum(image,axis=-1)
   
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    
    num_white_pixels = np.count_nonzero(colors_summed > 620)
    ratio_white_pixels = num_white_pixels / num_pixels
    
    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])
    
    return ratio_white_pixels, green_concentration, blue_concentration



def select_k_best_regions(regions, k=20):
    """
    Args:
        regions               list           list of 2-component tuples first component the region, 
                                             second component the ratio of white pixels
                                             
        k                     int            number of regions to select
    """
    regions = [x for x in regions if x[3] > 180 and x[4] > 180]
   
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return k_best_regions


def generate_patches(image, window_size=200, stride=128, k=20):
    
    colors_summed = np.sum(image,axis=-1)
    
    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    i = 0
    
    print_counter = 0

    
    while window_size + stride*i <= max_height:
        j = 0
        
        while window_size + stride*j <= max_width:   
      
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride
            
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            
            color_summed_patch = colors_summed[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size]
            
            
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch,color_summed_patch)
            
            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
            regions_container.append(region_tuple)
            
            j += 1
        
        i += 1
    
    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
    
    return k_best_region_coordinates, k_best_regions


def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x+window_size, y : y+window_size, :]
    
    return regions



def glue_to_one_picture(image_patches, window_size=200, k=16):
    side = int(np.sqrt(k))
    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)
        
    for i, patch in image_patches.items():
        x = i // side
        y = i % side
        image[
            x * window_size : (x+1) * window_size,
            y * window_size : (y+1) * window_size,
            :
        ] = patch
    
    return image

# test patch extraction 

import random

WINDOW_SIZE = 128
STRIDE = 64
K = 16

def test_patch_extraction(): 
    fig, ax = plt.subplots(6, 2, figsize=(20, 25))

    train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=10, random_state=random.seed())

    images = list(train_df['image_id'])
    labels = list(train_df['isup_grade'])

    data_dir = '../input/prostate-cancer-grade-assessment/train_images/'
    
    

    for i, img in enumerate(images[:6]):
        
        print("image number ",i)
        print("gc count",gc.get_count())
  
        url = data_dir + img + '.tiff'
        image = skimage.io.MultiImage(url)[-1]

        best_coordinates, best_regions = generate_patches(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K)
           
        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
        
        ax[i][0].imshow(image)
        ax[i][0].set_title(f'{img} - Original - Label: {labels[i]}')

        ax[i][1].imshow(glued_image)
        ax[i][1].set_title(f'{img} - Glued - Label: {labels[i]}')
        

    fig.suptitle('From biopsy to glued patches')
def get_patch_image(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K):
    best_coordinates, best_regions = generate_patches(image, window_size, stride, k)
    glued_image = glue_to_one_picture(best_regions, window_size, k)
    return glued_image
from tqdm.notebook import tqdm
import zipfile



destination_dir = '/patch_images/'
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

data_dir='/kaggle/input/prostate-cancer-grade-assessment/train_images/'

zipfile = '/kaggle/input/patch-images/patch_images.zip'


def get_image(data_dir,image_id):
    raw_image_url = data_dir + image_id +'.tiff'
    image = skimage.io.MultiImage(raw_image_url)[-1]
    return image


def create_patch_images(data_dir,destination_dir):
    for i in tqdm(range(train.shape[0])):
        image_id = train['image_id'].iloc[i]
        image = get_image(data_dir,image_id)
        patch_image = get_patch_image(image)
        patch_image_url = destination_dir + image_id + '.png'
        skimage.io.imsave(patch_image_url,patch_image,check_contrast=False)
        if (i % 100 == 0):
            gc.collect()
            #print(psutil.virtual_memory())
        
        
    shutil.make_archive('/patch_images/','zip','.')
        
def read_patch_images(zipfile):
    with zipfile.ZipFile(zipfile, 'r') as zip_ref:
        zip_ref.extractall('/patch_images/')
        
        

if os.path.exists(zipfile):
    read_patch_images()
else: 
    create_patch_images(data_dir,destination_dir)
    
print(psutil.virtual_memory())
    
labels=[]
data=[]
data_dir='/patch_images/'
for i in range(train.shape[0]):
    data.append(data_dir + train['image_id'].iloc[i]+'.png')
    labels.append(train['isup_grade'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['isup_grade']=labels
df
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['isup_grade'], test_size=0.2, random_state=1234)

train=pd.DataFrame(X_train)
train.columns=['images']
train['isup_grade']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['isup_grade']=y_val

train['isup_grade']=train['isup_grade'].astype(str)
validation['isup_grade']=validation['isup_grade'].astype(str)
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True)

val_datagen=train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validate_filenames=False)

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validate_filenames=False)
def vgg16_model( num_classes=None):

    model = VGG16(weights='/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
    x=Flatten()(model.output)
    output=Dense(num_classes,activation='softmax')(x)
    model=Model(model.input,output)
    return model

vgg_conv=vgg16_model(6)
def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)
opt = SGD(lr=0.001)
vgg_conv.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy',kappa_score])
nb_epochs = 3
batch_size = 16
nb_train_steps = train.shape[0]//batch_size
nb_val_steps=validation.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))

check_point = ModelCheckpoint('./model.h5',monitor='val_loss',verbose=True, save_best_only=True, save_weights_only=True)

early_stop = EarlyStopping(monitor='val_loss',patience=25,verbose=True)

callbacks = [check_point,early_stop]
vgg_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_val_steps,
    callbacks=callbacks,
    use_multiprocessing=True)
from functools import reduce
import skimage.io
from skimage.io import imshow,show
from skimage.transform import resize
from skimage.util import montage
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import skimage.transform

def predict_submission(df, path):
    
    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_id"]]
    df["isup_grade"] = 0
    predictions = []
    for idx, row in df.iterrows():
        image = skimage.io.imread(str(row.image_path))
        patch_image = get_patch_image(image)
        img = skimage.transform.resize(patch_image, (224,224))
        img = img.astype(np.float32)/255.
        img = np.reshape(img,(1,224,224,3))
        prediction = vgg_conv.predict(img)
        predictions.append(np.argmax(prediction))
            
    df["isup_grade"] = predictions
    df = df.drop('image_path', 1)
    return df[["image_id","isup_grade"]]
test_path = "../input/prostate-cancer-grade-assessment/train_images/"
submission_df = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")

if os.path.exists(test_path):
    test_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
    submission_df = predict_submission(test_df, test_path)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()
