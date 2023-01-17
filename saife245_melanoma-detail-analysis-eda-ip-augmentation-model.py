import os

import gc

import json

import math

import cv2

import PIL

import re

import numpy as np

import pandas as pd

from PIL import Image

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

#from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

%matplotlib inline

#from keras.preprocessing import image

import glob

import tensorflow.keras.applications.densenet as dense

from kaggle_datasets import KaggleDatasets

import seaborn as sns
tf.__version__
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')



print('Train: ', train.shape)

print("Test:", test.shape)
train.head()
train.info()
test.head()
test.info()
train['benign_malignant'].value_counts(normalize=True)
sns.countplot(train['benign_malignant'])
train['sex'].value_counts(normalize=True)
train['target'].groupby(train['sex']).mean()
sns.countplot(train['sex'], hue=train['target'])
train['target'].groupby(train['age_approx']).mean()
plt.figure(figsize=(8,5))

sns.countplot(train['age_approx'], hue=train['target'])
train['anatom_site_general_challenge'].value_counts(normalize=True)
train['target'].groupby(train['anatom_site_general_challenge']).mean()
plt.figure(figsize=(10,5))

sns.countplot(train['anatom_site_general_challenge'], hue=train['target'])
train['diagnosis'].value_counts(normalize=True)
train['target'].groupby(train['diagnosis']).mean()
plt.figure(figsize=(15,5))

sns.countplot(train['diagnosis'], hue=train['target'])
train_df = train[['sex','age_approx','anatom_site_general_challenge','diagnosis','target']]

train_df.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df = train_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

train_df.head()
g = sns.pairplot(train_df, hue="diagnosis")
sns.heatmap(train_df.corr(),annot=True,linewidths=0.2) 

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
def display_training_curves(training, validation, title, subplot):

  if subplot%10==1: # set up the subplots on the first call

    plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

    plt.tight_layout()

  ax = plt.subplot(subplot)

  ax.set_facecolor('#F8F8F8')

  ax.plot(training)

  ax.plot(validation)

  ax.set_title('model '+ title)

  ax.set_ylabel(title)

  ax.set_xlabel('epoch')

  ax.legend(['train', 'valid.'])



cols, rows = 4, 3

def grid_display(list_of_images, no_of_columns=2, figsize=(15,15), title = False):

    fig = plt.figure(figsize=figsize)

    column = 0

    z = 0

    for i in range(len(list_of_images)):

        column += 1

        #  check for end of column and create a new figure

        if column == no_of_columns+1:

            fig = plt.figure(figsize=figsize)

            column = 1

        fig.add_subplot(1, no_of_columns, column)

        if title:

            if i >= no_of_columns:

                plt.title(titles[z])

                z +=1

            else:

                plt.title(titles[i])

        plt.imshow(list_of_images[i])

        plt.axis('off')
image_list = train[train['target'] == 0].sample(8)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

#show_images(image_all, cols=1)

grid_display(image_all, 4, (15,15))
image_list = train[train['target'] == 1].sample(8)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'torso'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'lower extremity'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'upper extremity'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'head/neck'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'palms/soles'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['anatom_site_general_challenge'] == 'oral/genital'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'nevus'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'melanoma'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'seborrheic keratosis'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'lentigo NOS'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'lichenoid keratosis'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'solar lentigo'].sample(4)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'atypical melanocytic proliferation'].sample(1)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'cafe-au-lait macule'].sample(1)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
image_list = train[train['diagnosis'] == 'unknown'].sample()['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)

grid_display(image_all, 4, (15,15))
arr = [15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0]

image_all=[]

titles = ['At Age 15.0','At Age 20.0','At Age 25.0','At Age 30.0','At Age 35.0','At Age 40.0'

          ,'At Age 45.0','At Age 50.0','At Age 55.0','At Age 60.0','At Age 65.0','At Age 70.0'

          ,'At Age 75.0','At Age 80.0','At Age 85.0','At Age 90.0']

for i in arr:

    image_list = train[train['age_approx'] == i].sample()['image_name']

    for image_id in image_list:

        image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

        img = np.array(Image.open(image_file))

        image_all.append(img)

grid_display(image_all, 4, (15,15), title = True)
image_list = train[train['target'] == 1].sample(2)['image_name']

image_all=[]

titles = ['original', 'Reduced Noise', "Gaussian Blur", 'Adjusted Contrast']

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg'

    img = cv2.imread(image_file,1)

    image_all.append(img)

    #Reducing Noise

    result = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

    image_all.append(result)

    #Gaussian Blur

    blur_image = cv2.GaussianBlur(img, (7,7), 0)

    image_all.append(blur_image)

    #Adjusted contrast

    contrast_img = cv2.addWeighted(img, 2.5, np.zeros(img.shape, img.dtype), 0, 0)

    image_all.append(contrast_img)

grid_display(image_all, 4, (15,15), title = True)
image_list = train[train['target'] == 1].sample(2)['image_name']

image_all=[]

titles = ['original', 'Adaptive thresholding', "Binary thresholding"]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg'

    img = cv2.imread(image_file,1)

    image_all.append(img)

    #Adaptive Thresholding..

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    image_all.append(thresh1)

    #Binary Thresholding...

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    res, thresh = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV)

    image_all.append(thresh)

grid_display(image_all, 3, (15,15), title = True)
img = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0052212.jpg', 0)

# global thresholding

ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)



# Otsu's thresholding

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(5,5),0)

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# plot all the images and their histograms

images = [img, 0, th1,

          img, 0, th2,

          blur, 0, th3]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',

          'Original Noisy Image','Histogram',"Otsu's Thresholding",

          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

plt.figure(figsize=(15,10))

for i in range(3):

    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')

    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)

    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')

    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()
image = []

titles = ['Original', 'Thresold Image', 'Contour Image']

img = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0052212.jpg', 0)

img = img[200:900, 500:1500]

image.append(img)

#Apply thresholding

blur = cv2.GaussianBlur(img,(5,5),0)

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

image.append(th3)

contours, hierarcy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



img2 = img.copy()

index = -1

thickness = 4

color = (255, 0, 255)



objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')

for c in contours:

    cv2.drawContours(objects, [c], -1, color, -1)

    

    area = cv2.contourArea(c)

    perimeter = cv2.arcLength(c, True)

    

    M = cv2.moments(c)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])

        cy = int(M["m01"] / M["m00"])

    else:

    # set values as what you need in the situation

        cx, cy = 0, 0

    cv2.circle(objects, (cx, cy), 4, (0, 0, 255), -1)

    

    print("AREA:{}, perimeter:{}".format(area, perimeter))



image.append(objects)

grid_display(image, 3, (15,15), title = True)
image_list = train[train['target'] == 1].sample(2)['image_name']

image_all=[]

titles = ['original', 'Scale Down', "Scale Up"]

for image_id in image_list:

    scaleX = 0.6

    scaleY = 0.6

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg'

    img = cv2.imread(image_file,1)

    image_all.append(img)

    # Scaling Down the image 0.6 times

    scaleDown = cv2.resize(img, None, fx= scaleX, fy= scaleY, interpolation= cv2.INTER_LINEAR)

    image_all.append(scaleDown)

    # Scaling up the image 1.8 times

    scaleUp = cv2.resize(img, None, fx= scaleX*3, fy= scaleY*3, interpolation= cv2.INTER_LINEAR)

    image_all.append(scaleUp)

grid_display(image_all, 3, (15,15), title = True)
image_all=[]

titles = ['original', 'ORB Detected', "Zoom Image"]

img = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0052212.jpg', 1)

image_all.append(img)

# Initiate ORB detector

orb = cv2.ORB_create()

# find the keypoints with ORB

kp = orb.detect(img,None)

# compute the descriptors with ORB

kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation

img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

image_all.append(img2)

img3 = img2[350:800,600:1250]

image_all.append(img3)

grid_display(image_all, 3, (35,35), title = True)
# Detect hardware

try:

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection

except ValueError:

  tpu = None

#If TPU not found try with GPUs

  gpus = tf.config.experimental.list_logical_devices("GPU")

    

# Select appropriate distribution strategy for hardware

if tpu:

  tf.config.experimental_connect_to_cluster(tpu)

  tf.tpu.experimental.initialize_tpu_system(tpu)

  strategy = tf.distribute.experimental.TPUStrategy(tpu)

  print('Running on TPU ', tpu.master())  

elif len(gpus) > 0:

  strategy = tf.distribute.MirroredStrategy(gpus) # this works for 1 to multiple GPUs

  print('Running on ', len(gpus), ' GPU(s) ')

else:

  strategy = tf.distribute.get_strategy()

  print('Running on CPU')



# How many accelerators do we have ?

print("Number of accelerators: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/train*')

TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')

BATCH_SIZE = 10 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]

AUTO = tf.data.experimental.AUTOTUNE

imSize = 1024

EPOCHS = 10



VALIDATION_SPLIT = 0.18

split = int(len(TRAINING_FILENAMES) * VALIDATION_SPLIT)

training_filenames = TRAINING_FILENAMES[split:]

validation_filenames = TRAINING_FILENAMES[:split]

print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files"

      .format(len(TRAINING_FILENAMES), len(training_filenames), len(validation_filenames)))

TRAINING_FILENAMES = training_filenames
def read_labeled_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    image = tf.image.resize(image, [imSize,imSize])

    label = tf.cast(example['target'], tf.int32)

    return image, label 



def read_unlabeled_tfrecord(example):

    u_features = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, u_features)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    image = tf.image.resize(image, [imSize,imSize])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset
def data_augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    #image = tf.image.random_brightness(x, 0.2)

    #image = cutmix(image, label)

    return image, label   



def get_training_dataset(dataset, do_aug=True):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(filenames, train=False):

    dataset = load_dataset(filenames, labeled=True)

    dataset = dataset.cache() # This dataset fits in RAM

    if train:

    # Best practices for Keras:

    # Training dataset: repeat then batch

    # Evaluation dataset: do not repeat

        dataset = dataset.repeat()

        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

        dataset = dataset.shuffle(2000)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset

    

def get_test_dataset(dataset, ordered=False):

    dataset = load_dataset(dataset, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



validation_dataset = get_validation_dataset(validation_filenames, train=False)

training_dataset = get_training_dataset(TRAINING_FILENAMES)

test_dataset = get_test_dataset(TEST_FILENAMES)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

NUM_VALID_IMAGES = count_data_items(validation_filenames)

validation_steps = NUM_VALID_IMAGES // BATCH_SIZE

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images, {} validition images'

      .format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES, NUM_VALID_IMAGES))
def get_random_eraser(input_img,p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

   # def eraser(input_img):

    img_h, img_w, img_c = input_img.shape



    while True:

        s = np.random.uniform(s_l, s_h) * img_h * img_w

        r = np.random.uniform(r_1, r_2)

        w = int(np.sqrt(s / r))

        h = int(np.sqrt(s * r))

        left = np.random.randint(0, img_w)

        top = np.random.randint(0, img_h)



        if left + w <= img_w and top + h <= img_h:

            break



    if pixel_level:

        c = np.random.uniform(v_l, v_h, (h, w, img_c))

    else:

        c = np.random.uniform(v_l, v_h)



    input_img[top:top + h, left:left + w, :] = c



    return input_img
TRAIN = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

IMAGE_SIZE = 1024

n_imgs = 12

img_filenames = os.listdir(TRAIN)[:n_imgs]

img_filenames[:3]

image=[]

for file_name in img_filenames:

    img = cv2.imread(TRAIN +file_name)

    img = get_random_eraser(img)

    image.append(img)

grid_display(image, 4, (15,15))
# if you have label in images

def onehot(image,label):

    CLASSES = 2 # Define number of classes our model have

    return image,tf.one_hot(label,CLASSES)



def cutmix(image, label): #, PROBABILITY = 1.0

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with cutmix applied

    DIM = 1024 #IMAGE_SIZE[0]

    CLASSES = 2

    

    imgs = []; labs = []

    for j in range(AUG_BATCH):

        # CHOOSE RANDOM IMAGE TO CUTMIX WITH

        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)

        # CHOOSE RANDOM LOCATION

        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32)

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        # MAKE CUTMIX IMAGE

        one = image[j,ya:yb,0:xa,:]

        two = image[k,ya:yb,xa:xb,:]

        three = image[j,ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)

        imgs.append(img)

        # MAKE CUTMIX LABEL

        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)

        if len(label.shape)==1:

            lab1 = tf.one_hot(label[j],CLASSES)

            lab2 = tf.one_hot(label[k],CLASSES)

        else:

            lab1 = label[j,]

            lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image2,label2
AUG_BATCH = 48

row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = training_dataset.unbatch()

augmented_element = all_elements.repeat().batch(AUG_BATCH).map(cutmix)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
def transform(image, inv_mat, image_shape):

    h, w, c = image_shape

    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)

    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])

    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))

    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)

    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)

    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))

    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))

    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))

    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)

    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)

    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))

    rotated_image_channel = list()

    for i in range(c):

        vals = rotated_image_values[:,i]

        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])

        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])



def random_rotate(image, angle, image_shape):

    def get_rotation_mat_inv(angle):

        # transform to radian

        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)

        sin_val = tf.math.sin(angle)

        one = tf.constant([1], tf.float32)

        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)

        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv

    angle = float(angle) * tf.random.normal([1],dtype='float32')

    rot_mat_inv = get_rotation_mat_inv(angle)

    return transform(image, rot_mat_inv, image_shape)





def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    h, w = image_height, image_width

    hh = int(np.ceil(np.sqrt(h*h+w*w)))

    hh = hh+1 if hh%2==1 else hh

    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)

    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)



    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)



    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)

    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)



    for i in range(0, hh//d+1):

        s1 = i * d + st_h

        s2 = i * d + st_w

        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)

        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)



    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)

    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)

    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)



    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))

    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))



    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])

    x_ranges = tf.repeat(x_ranges, hh)

    y_ranges = tf.repeat(y_ranges, hh)



    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))

    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))



    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])

    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)



    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])

    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)



    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)



    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])

    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)



    return mask



def apply_grid_mask(image, image_shape):

    AugParams = {

        'd1' : 100,

        'd2': 160,

        'rotate' : 45,

        'ratio' : 0.3

    }

    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])

    if image_shape[-1] == 3:

        mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask,tf.float32)



def gridmask(img_batch, label_batch):

    return apply_grid_mask(img_batch, (1024,1024, 3)), label_batch
AUG_BATCH = 48

row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = training_dataset.unbatch()

augmented_element = all_elements.repeat().batch(AUG_BATCH).map(gridmask)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
# data dump

print("Training data shapes:")

for image, label in training_dataset.take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())



print("Validation data shapes:")

for image, label in validation_dataset.take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())



print("Test data shapes:")

for image, idnum in test_dataset.take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U'))
# with strategy.scope():

#     model = tf.keras.Sequential([

#         dense.DenseNet121(

#             input_shape=(imSize, imSize, 3),

#             weights='imagenet',

#             include_top=False

#         ),

#         layers.GlobalAveragePooling2D(),

#         layers.Dense(1, activation='sigmoid')

#     ])

        

#     model.compile(

#         optimizer='adam',

#         loss = 'binary_crossentropy',

#         metrics=['accuracy']

#     )

#     model.summary()
import tensorflow.keras.applications.xception as xcep

with strategy.scope():

    model = tf.keras.Sequential([

        xcep.Xception(

            input_shape=(imSize, imSize, 3),

            weights='imagenet',

            include_top=False

        ),

        layers.GlobalAveragePooling2D(),

        layers.Dense(1024, activation= 'relu'), 

        layers.Dropout(0.2),

        layers.Dense(512, activation= 'relu'), 

        layers.Dropout(0.2), 

        layers.Dense(256, activation='relu'), 

        layers.Dropout(0.2), 

        layers.Dense(128, activation='relu'), 

        layers.Dropout(0.1),

        layers.Dense(64, activation='relu'), 

        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')

    ])

        

    model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=['accuracy']

    )

    model.summary()
# !pip install -q efficientnet
# import efficientnet.tfkeras as efn

# with strategy.scope():

#     model = tf.keras.Sequential([

#         efn.EfficientNetB5(

#             input_shape=(imSize, imSize, 3),

#             weights='imagenet',

#             include_top=False

#         ),

#         layers.GlobalAveragePooling2D(),

#         layers.Dense(512, activation= 'relu'), 

#         layers.Dropout(0.2), 

#         layers.Dense(256, activation='relu'), 

#         layers.Dropout(0.2), 

#         layers.Dense(128, activation='relu'), 

#         layers.Dropout(0.1),

#         layers.Dense(64, activation='relu'), 

#         layers.Dropout(0.1),

#         layers.Dense(1, activation='sigmoid')

#     ])

        

#     model.compile(

#         optimizer='adam',

#         loss = 'binary_crossentropy',

#         metrics=['accuracy']

#     )

#     model.summary()
def lrfn(epoch):

    LR_START = 0.00001

    LR_MAX = 0.00005 * strategy.num_replicas_in_sync

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

EPOCHS = 10

history = model.fit(training_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,

                    validation_data=validation_dataset,callbacks=[lr_schedule])
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)
test_ds = test_dataset#get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds).flatten()

print(probabilities)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, probabilities]), fmt=['%s', '%f'],

           delimiter=',', header='image_name,target', comments='')

sub = pd.read_csv("submission.csv")

sub.head()
sub['target'].hist()