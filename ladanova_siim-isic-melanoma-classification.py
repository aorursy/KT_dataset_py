import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl

from IPython import display

import seaborn as sns

sns.set()



import re



import pydicom

import random



import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models



#import pytorch_lightning as pl

from scipy.special import softmax



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import roc_auc_score, auc



from skimage.io import imread

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.subplots import make_subplots



import os

import copy



from albumentations import Compose, RandomCrop, Normalize,HorizontalFlip, Resize

from albumentations import VerticalFlip, RGBShift, RandomBrightness

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.pytorch import ToTensor



from tqdm.notebook import tqdm



os.listdir("../input/")
basepath = "../input/siim-isic-melanoma-classification/"

modelspath = "../input/pytorch-pretrained-image-models/"

imagestatspath = "../input/siimisic-melanoma-classification-image-stats/"

foldspath = "../input/irndropout/"
os.listdir(basepath)
train_info = pd.read_csv(basepath + "train.csv")

train_info.head()
test_info = pd.read_csv(basepath + "test.csv")

test_info.head()
train_info.shape[0] / test_info.shape[0]
missing_vals_train = train_info.isnull().sum() / train_info.shape[0]

missing_vals_train[missing_vals_train > 0].sort_values(ascending=False)
missing_vals_test = test_info.isnull().sum() / test_info.shape[0]

missing_vals_test[missing_vals_test > 0].sort_values(ascending=False)
train_info.image_name.value_counts().max()
test_info.image_name.value_counts().max()
train_info.patient_id.value_counts().max()
test_info.patient_id.value_counts().max()
patient_counts_train = train_info.patient_id.value_counts()

patient_counts_test = test_info.patient_id.value_counts()



fig, ax = plt.subplots(2,2,figsize=(20,12))



sns.distplot(patient_counts_train, ax=ax[0,0], color="orangered", kde=True);

ax[0,0].set_xlabel("Counts")

ax[0,0].set_ylabel("Frequency")

ax[0,0].set_title("Patient id value counts in train");



sns.distplot(patient_counts_test, ax=ax[0,1], color="lightseagreen", kde=True);

ax[0,1].set_xlabel("Counts")

ax[0,1].set_ylabel("Frequency")

ax[0,1].set_title("Patient id value counts in test");



sns.boxplot(patient_counts_train, ax=ax[1,0], color="orangered");

ax[1,0].set_xlim(0, 250)

sns.boxplot(patient_counts_test, ax=ax[1,1], color="lightseagreen");

ax[1,1].set_xlim(0, 250);
np.quantile(patient_counts_train, 0.75) - np.quantile(patient_counts_train, 0.25)
np.quantile(patient_counts_train, 0.5)
print(np.quantile(patient_counts_train, 0.95))

print(np.quantile(patient_counts_test, 0.95))
200/test_info.shape[0] * 100
train_patient_ids = set(train_info.patient_id.unique())

test_patient_ids = set(test_info.patient_id.unique())



train_patient_ids.intersection(test_patient_ids)
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_info.sex, palette="Reds_r", ax=ax[0]);

ax[0].set_xlabel("")

ax[0].set_title("Gender counts");



sns.countplot(test_info.sex, palette="Blues_r", ax=ax[1]);

ax[1].set_xlabel("")

ax[1].set_title("Gender counts");
fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.countplot(train_info.age_approx, color="orangered", ax=ax[0]);

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_xlabel("");

ax[0].set_title("Age distribution in train");



sns.countplot(test_info.age_approx, color="lightseagreen", ax=ax[1]);

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_xlabel("");

ax[1].set_title("Age distribution in test");
fig, ax = plt.subplots(1,2,figsize=(20,5))



image_locations_train = train_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)

image_locations_test = test_info.anatom_site_general_challenge.value_counts().sort_values(ascending=False)



sns.barplot(x=image_locations_train.index.values, y=image_locations_train.values, ax=ax[0], color="orangered");

ax[0].set_xlabel("");

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_title("Image locations in train");



sns.barplot(x=image_locations_test.index.values, y=image_locations_test.values, ax=ax[1], color="lightseagreen");

ax[1].set_xlabel("");

labels = ax[1].get_xticklabels();

ax[1].set_xticklabels(labels, rotation=90);

ax[1].set_title("Image locations in test");
# Target distribution

fig, ax = plt.subplots(1,2, figsize=(20,5))



sns.countplot(x=train_info.diagnosis, orient="v", ax=ax[0], color="Orangered")

ax[0].set_xlabel("")

labels = ax[0].get_xticklabels();

ax[0].set_xticklabels(labels, rotation=90);

ax[0].set_title("Diagnosis");



sns.countplot(train_info.benign_malignant, ax=ax[1], palette="Reds_r");

ax[1].set_xlabel("")

ax[1].set_title("Type");
# Training Data Grouped by benign_malignant

train_info.groupby("benign_malignant").target.nunique()
patient_ages_table_train = train_info.groupby(["patient_id", "age_approx"]).size() / train_info.groupby("patient_id").size()

patient_ages_table_train = patient_ages_table_train.unstack().transpose()

patient_ages_table_test = test_info.groupby(["patient_id", "age_approx"]).size() / test_info.groupby("patient_id").size()

patient_ages_table_test = patient_ages_table_test.unstack().transpose()



patient_with_known_ages_train = train_info[train_info.patient_id.isin(patient_ages_table_train.columns.values)]



sorted_patients_train = patient_with_known_ages_train.patient_id.value_counts().index.values

patient_with_known_ages_test = test_info[test_info.patient_id.isin(patient_ages_table_test.columns.values)]

sorted_patients_test = patient_with_known_ages_test.patient_id.value_counts().index.values



fig, ax = plt.subplots(2,1, figsize=(20,20))

sns.heatmap(patient_ages_table_train[sorted_patients_train], cmap="Reds", ax=ax[0], cbar=False);

ax[0].set_title("Image coverage in % per patient and age in train data");

sns.heatmap(patient_ages_table_test[sorted_patients_test], cmap="Blues", ax=ax[1], cbar=False);

ax[1].set_title("Image coverage in % per patient and age in test data");

ax[0].set_xlabel("")

ax[1].set_xlabel("");
fig, ax = plt.subplots(2,2,figsize=(20,15))



sns.boxplot(train_info.sex, train_info.age_approx, ax=ax[0,0], palette="Reds_r");

ax[0,0].set_title("Age per gender in train");



sns.boxplot(test_info.sex, test_info.age_approx, ax=ax[0,1], palette="Blues_r");

ax[0,1].set_title("Age per gender in test");



sns.countplot(train_info.age_approx, hue=train_info.sex, ax=ax[1,0], palette="Reds_r");

sns.countplot(test_info.age_approx, hue=test_info.sex, ax=ax[1,1], palette="Blues_r");
sex_and_cancer_map = train_info.groupby(

    ["benign_malignant", "sex"]

).size().unstack(level=0) / train_info.groupby("benign_malignant").size() * 100



cancer_sex_map = train_info.groupby(

    ["benign_malignant", "sex"]

).size().unstack(level=1) / train_info.groupby("sex").size() * 100





fig, ax = plt.subplots(1,3,figsize=(20,5))



sns.boxplot(train_info.benign_malignant, train_info.age_approx, ax=ax[0], palette="Greens");

ax[0].set_title("Age and cancer");

ax[0].set_xlabel("");



sns.heatmap(sex_and_cancer_map, annot=True, cmap="Greens", cbar=False, ax=ax[1])

ax[1].set_xlabel("")

ax[1].set_ylabel("");



sns.heatmap(cancer_sex_map, annot=True, cmap="Greens", cbar=False, ax=ax[2])

ax[2].set_xlabel("")

ax[2].set_ylabel("");
fig, ax = plt.subplots(2,2,figsize=(20,15))



sns.countplot(train_info[train_info.benign_malignant=="benign"].age_approx, hue=train_info.sex, palette="Purples_r", ax=ax[0,0])

ax[0,0].set_title("Benign cases in train");



sns.countplot(train_info[train_info.benign_malignant=="malignant"].age_approx, hue=train_info.sex, palette="Oranges_r", ax=ax[0,1])

ax[0,1].set_title("Malignant cases in train");



sns.violinplot(train_info.sex, train_info.age_approx, hue=train_info.benign_malignant, split=True, ax=ax[1,0], palette="Greens_r");

sns.violinplot(train_info.benign_malignant, train_info.age_approx, hue=train_info.sex, split=True, ax=ax[1,1], palette="RdPu");
patient_gender_train = train_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])

patient_gender_test = test_info.groupby("patient_id").sex.unique().apply(lambda l: l[0])



train_patients = pd.DataFrame(index=patient_gender_train.index.values, data=patient_gender_train.values, columns=["sex"])

test_patients = pd.DataFrame(index=patient_gender_test.index.values, data=patient_gender_test.values, columns=["sex"])



train_patients.loc[:, "num_images"] = train_info.groupby("patient_id").size()

test_patients.loc[:, "num_images"] = test_info.groupby("patient_id").size()



train_patients.loc[:, "min_age"] = train_info.groupby("patient_id").age_approx.min()

train_patients.loc[:, "max_age"] = train_info.groupby("patient_id").age_approx.max()

test_patients.loc[:, "min_age"] = test_info.groupby("patient_id").age_approx.min()

test_patients.loc[:, "max_age"] = test_info.groupby("patient_id").age_approx.max()



train_patients.loc[:, "age_span"] = train_patients["max_age"] - train_patients["min_age"]

test_patients.loc[:, "age_span"] = test_patients["max_age"] - test_patients["min_age"]



train_patients.loc[:, "benign_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "benign"]

train_patients.loc[:, "malignant_cases"] = train_info.groupby(["patient_id", "benign_malignant"]).size().loc[:, "malignant"]

train_patients["min_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.min().loc[:, "malignant"]

train_patients["max_age_malignant"] = train_info.groupby(["patient_id", "benign_malignant"]).age_approx.max().loc[:, "malignant"]
train_patients.sort_values(by="malignant_cases", ascending=False).head()
fig, ax = plt.subplots(2,2,figsize=(20,12))

sns.countplot(train_patients.sex, ax=ax[0,0], palette="Reds")

ax[0,0].set_title("Gender counts with unique patient ids in train")

sns.countplot(test_patients.sex, ax=ax[0,1], palette="Blues");

ax[0,1].set_title("Gender counts with unique patient ids in test");



train_age_span_perc = train_patients.age_span.value_counts() / train_patients.shape[0] * 100

test_age_span_perc = test_patients.age_span.value_counts() / test_patients.shape[0] * 100



sns.barplot(train_age_span_perc.index, train_age_span_perc.values, ax=ax[1,0], color="Orangered");

sns.barplot(test_age_span_perc.index, test_age_span_perc.values, ax=ax[1,1], color="Lightseagreen");

ax[1,0].set_title("Patients age span in train")

ax[1,1].set_title("Patients age span in test")

for n in range(2):

    ax[1,n].set_ylabel("% in data")

    ax[1,n].set_xlabel("age span");
example_files = os.listdir(basepath + "train/")[0:2]

example_files
train_info.head()
train_info["dcm_path"] = basepath + "train/" + train_info.image_name + ".dcm"

test_info["dcm_path"] = basepath + "test/" + test_info.image_name + ".dcm"
print(train_info.dcm_path[0])

print(test_info.dcm_path[0])
example_dcm = pydicom.dcmread(train_info.dcm_path[2])

example_dcm
image = example_dcm.pixel_array

print(image.shape)
train_info["image_path"] = basepath + "jpeg/train/" + train_info.image_name + ".jpg"

test_info["image_path"] = basepath + "jpeg/test/" + test_info.image_name + ".jpg"
os.listdir(imagestatspath)
test_image_stats = pd.read_csv(imagestatspath +  "test_image_stats.csv")

test_image_stats.head(1)
train_image_stats_1 = pd.read_csv(imagestatspath + "train_image_stats_10000.csv")

train_image_stats_2 = pd.read_csv(imagestatspath + "train_image_stats_20000.csv")

train_image_stats_3 = pd.read_csv(imagestatspath + "train_image_stats_toend.csv")

train_image_stats_4 = train_image_stats_1.append(train_image_stats_2)

train_image_stats = train_image_stats_4.append(train_image_stats_3)

train_image_stats.shape
plot_test = True
if plot_test:

    N = test_image_stats.shape[0]

    selected_data = test_image_stats

    my_title = "Test image statistics"

else:

    N = train_image_stats.shape[0]

    selected_data = train_image_stats

    my_title = "Train image statistics"



trace1 = go.Scatter3d(

    x=selected_data.img_mean.values[0:N], 

    y=selected_data.img_std.values[0:N],

    z=selected_data.img_skew.values[0:N],

    mode='markers',

    text=selected_data["rows"].values[0:N],

    marker=dict(

        color=selected_data["columns"].values[0:N],

        colorscale = "Jet",

        colorbar=dict(thickness=10, title="image columns", len=0.8),

        opacity=0.4,

        size=2

    )

)

figure_data = [trace1]

layout = go.Layout(

    title = my_title,

    scene = dict(

        xaxis = dict(title="Image mean"),

        yaxis = dict(title="Image standard deviation"),

        zaxis = dict(title="Image skewness"),

    ),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    showlegend=True

)



fig = go.Figure(data=figure_data, layout=layout)

py.iplot(fig, filename='simple-3d-scatter')
test_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / test_image_stats.shape[0]
train_image_stats.groupby(["rows", "columns"]).size().sort_values(ascending=False).iloc[0:10] / train_image_stats.shape[0]
examples1 = {"rows": 1080, "columns": 1920}

examples2 = {"rows": 4000, "columns": 6000}
selection1 = np.random.choice(test_image_stats[

    (test_image_stats["rows"]==examples1["rows"]) & (test_image_stats["columns"]==examples1["columns"])

].path.values, size=8, replace=False)



fig, ax = plt.subplots(2,4,figsize=(20,8))



for n in range(2):

    for m in range(4):

        path = selection1[m + n*4]

        dcm_file = pydicom.dcmread(path)

        image = dcm_file.pixel_array

        ax[n,m].imshow(image)

        ax[n,m].grid(False)
selection2 = np.random.choice(test_image_stats[

    (test_image_stats["rows"]==examples2["rows"]) & (test_image_stats["columns"]==examples2["columns"])

].path.values, size=8, replace=False)



fig, ax = plt.subplots(2,4,figsize=(20,6))



for n in range(2):

    for m in range(4):

        path = selection2[m + n*4]

        dcm_file = pydicom.dcmread(path)

        image = dcm_file.pixel_array

        ax[n,m].imshow(image)

        ax[n,m].grid(False)
import cv2, pandas as pd, matplotlib.pyplot as plt

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

print('Examples WITH Melanoma')

imgs = train.loc[train.target==1].sample(10).image_name.values

plt.figure(figsize=(20,8))

for i,k in enumerate(imgs):

    img = cv2.imread('../input/jpeg-melanoma-128x128/train/%s.jpg'%k)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.subplot(2,5,i+1); plt.axis('off')

    plt.imshow(img)

plt.show()

print('Examples WITHOUT Melanoma')

imgs = train.loc[train.target==0].sample(10).image_name.values

plt.figure(figsize=(20,8))

for i,k in enumerate(imgs):

    img = cv2.imread('../input/jpeg-melanoma-128x128/train/%s.jpg'%k)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.subplot(2,5,i+1); plt.axis('off')

    plt.imshow(img)

plt.show()
import pandas as pd, numpy as np

from kaggle_datasets import KaggleDatasets

import tensorflow as tf, re, math

import tensorflow.keras.backend as K

import tensorflow.keras.applications.inception_resnet_v2 as irn

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
DEVICE = "TPU" #or "GPU"



# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD

SEED = 42



# NUMBER OF FOLDS. USE 3, 5, OR 15 

FOLDS = 5



# WHICH IMAGE SIZES TO LOAD EACH FOLD

# CHOOSE 128, 192, 256, 384, 512, 768 

IMG_SIZES = [256, 256, 256, 256, 256]



# INCLUDE OLD COMP DATA? YES=1 NO=0

INC2019 = [0,0,1,1,1]

INC2018 = [1,1,0,0,0]



# BATCH SIZE AND EPOCHS

BATCH_SIZES = [32]*FOLDS

EPOCHS = [12]*FOLDS



# WHICH EFFICIENTNET B? TO USE

EFF_NETS = [0,0,0,0,0]



# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST

WGTS = [1/FOLDS]*FOLDS



# TEST TIME AUGMENTATION STEPS

TTA = 11
if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS

for i,k in enumerate(IMG_SIZES):

    GCS_PATH[i] = KaggleDatasets().get_gcs_path('melanoma-%ix%i'%(k,k))

    GCS_PATH2[i] = KaggleDatasets().get_gcs_path('isic2019-%ix%i'%(k,k))

files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))

files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))
ROT_ = 180.0

SHR_ = 2.0

HZOOM_ = 8.0

WZOOM_ = 8.0

HSHIFT_ = 8.0

WSHIFT_ = 8.0
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear    = math.pi * shear    / 180.



    def get_3x3_mat(lst):

        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    

    # ROTATION MATRIX

    c1   = tf.math.cos(rotation)

    s1   = tf.math.sin(rotation)

    one  = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    

    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 

                                   -s1,  c1,   zero, 

                                   zero, zero, one])    

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)    

    

    shear_matrix = get_3x3_mat([one,  s2,   zero, 

                                zero, c2,   zero, 

                                zero, zero, one])        

    # ZOOM MATRIX

    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 

                               zero,            one/width_zoom, zero, 

                               zero,            zero,           one])    

    # SHIFT MATRIX

    shift_matrix = get_3x3_mat([one,  zero, height_shift, 

                                zero, one,  width_shift, 

                                zero, zero, one])

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), 

                 K.dot(zoom_matrix,     shift_matrix))





def transform(image, DIM=256):    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    XDIM = DIM%2 #fix for size 331

    

    rot = ROT_ * tf.random.normal([1], dtype='float32')

    shr = SHR_ * tf.random.normal([1], dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_

    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 

    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)

    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])

    z   = tf.ones([DIM*DIM], dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])

    d    = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM, DIM,3])
def read_labeled_tfrecord(example):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }           

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['target']





def read_unlabeled_tfrecord(example, return_image_name):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['image_name'] if return_image_name else 0



 

def prepare_image(img, augment=True, dim=256):    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.cast(img, tf.float32) / 255.0

    

    if augment:

        img = transform(img,DIM=dim)

        img = tf.image.random_flip_left_right(img)

        #img = tf.image.random_hue(img, 0.01)

        img = tf.image.random_saturation(img, 0.7, 1.3)

        img = tf.image.random_contrast(img, 0.8, 1.2)

        img = tf.image.random_brightness(img, 0.1)

                      

    img = tf.reshape(img, [dim,dim, 3])

            

    return img



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
def get_dataset(files, augment = False, shuffle = False, repeat = False, 

                labeled=True, return_image_names=True, batch_size=16, dim=256):

    

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)

        

    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 

                    num_parallel_calls=AUTO)      

    

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), 

                                               imgname_or_label), 

                num_parallel_calls=AUTO)

    

    ds = ds.batch(batch_size * REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds
EFNS = [irn.InceptionResNetV2]



def build_model(dim=128, ef=0):

    inp = tf.keras.layers.Input(shape=(dim,dim,3))

    base = EFNS[ef](input_shape=(dim,dim,3),weights='imagenet',include_top=False)

    x = base(inp)

    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp,outputs=x)

#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 

    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])

    return model

EFNS = [irn.InceptionResNetV2] # [efn.EfficientNetB7]    

def get_lr_callback(batch_size=8):

    lr_start   = 0.000005

    lr_max     = 0.00000125 * REPLICAS * batch_size

    lr_min     = 0.000001

    lr_ramp_ep = 5

    lr_sus_ep  = 0

    lr_decay   = 0.8

   

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            

        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max

            

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            

        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback
# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit

VERBOSE = 2

DISPLAY_PLOT = False



skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

oof_pred = []; oof_tar = []; oof_val = []; oof_names = []; oof_folds = [] 

preds = np.zeros((count_data_items(files_test),1))



for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

    

    # DISPLAY FOLD INFO

    if DEVICE=='TPU':

        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)

    print('#'*25); print('#### FOLD',fold+1)

    print('#### Image Size %i with InceptionResNetV2 and batch_size %i'%

          (IMG_SIZES[fold],BATCH_SIZES[fold]*REPLICAS))

    

    # CREATE TRAIN AND VALIDATION SUBSETS

    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])

    if INC2019[fold]:

        files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+1])

        print('#### Using 2019 external data')

    if INC2018[fold]:

        files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2])

        print('#### Using 2018+2017 external data')

    np.random.shuffle(files_train); print('#'*25)

    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])

    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))

    

    # BUILD MODEL

    K.clear_session()

    with strategy.scope():

        model = build_model(dim=IMG_SIZES[fold],ef=EFF_NETS[fold])

        

    # SAVE BEST MODEL EACH FOLD

    sv = tf.keras.callbacks.ModelCheckpoint(

        'fold-%i.h5'%fold, monitor='val_loss', verbose=0, save_best_only=True,

        save_weights_only=True, mode='min', save_freq='epoch')

   

    # TRAIN

    print('Training...')

    history = model.fit(

        get_dataset(files_train, augment=True, shuffle=True, repeat=True,

                dim=IMG_SIZES[fold], batch_size = BATCH_SIZES[fold]), 

        epochs=EPOCHS[fold], callbacks = [sv,get_lr_callback(BATCH_SIZES[fold])], 

        steps_per_epoch=count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,

        validation_data=get_dataset(files_valid,augment=False,shuffle=False,

                repeat=False,dim=IMG_SIZES[fold]), #class_weight = {0:1,1:2},

        verbose=VERBOSE

    )

    

    print('Loading best model...')

    model.load_weights('fold-%i.h5'%fold)

    

    # PREDICT OOF USING TTA

    print('Predicting OOF with TTA...')

    ds_valid = get_dataset(files_valid,labeled=False,return_image_names=False,augment=True,

            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4)

    ct_valid = count_data_items(files_valid); STEPS = TTA * ct_valid/BATCH_SIZES[fold]/4/REPLICAS

    pred = model.predict(ds_valid,steps=STEPS,verbose=VERBOSE)[:TTA*ct_valid,] 

    oof_pred.append( np.mean(pred.reshape((ct_valid,TTA),order='F'),axis=1) )                 

    #oof_pred.append(model.predict(get_dataset(files_valid,dim=IMG_SIZES[fold]),verbose=1))

    

    # GET OOF TARGETS AND NAMES

    ds_valid = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],

            labeled=True, return_image_names=True)

    oof_tar.append( np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]) )

    oof_folds.append( np.ones_like(oof_tar[-1],dtype='int8')*fold )

    ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],

                labeled=False, return_image_names=True)

    oof_names.append( np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

    

    # PREDICT TEST USING TTA

    print('Predicting Test with TTA...')

    ds_test = get_dataset(files_test,labeled=False,return_image_names=False,augment=True,

            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4)

    ct_test = count_data_items(files_test); STEPS = TTA * ct_test/BATCH_SIZES[fold]/4/REPLICAS

    pred = model.predict(ds_test,steps=STEPS,verbose=VERBOSE)[:TTA*ct_test,] 

    preds[:,0] += np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) * WGTS[fold]

    

    # REPORT RESULTS

    auc = roc_auc_score(oof_tar[-1],oof_pred[-1])

    oof_val.append(np.max( history.history['val_auc'] ))

    print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'%(fold+1,oof_val[-1],auc))

    

    # PLOT TRAINING

    if DISPLAY_PLOT:

        plt.figure(figsize=(15,5))

        plt.plot(np.arange(EPOCHS[fold]),history.history['auc'],'-o',label='Train AUC',color='#ff7f0e')

        plt.plot(np.arange(EPOCHS[fold]),history.history['val_auc'],'-o',label='Val AUC',color='#1f77b4')

        x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )

        xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]

        plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)

        plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)

        plt.legend(loc=2)

        plt2 = plt.gca().twinx()

        plt2.plot(np.arange(EPOCHS[fold]),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')

        plt2.plot(np.arange(EPOCHS[fold]),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')

        x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )

        ydist = plt.ylim()[1] - plt.ylim()[0]

        plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)

        plt.ylabel('Loss',size=14)

        plt.title('FOLD %i - Image Size %i, InceptionResNetV2, inc2019=%i, inc2018=%i'%

                (fold+1,IMG_SIZES[fold],INC2019[fold],INC2018[fold]),size=18)

        plt.legend(loc=3)

        plt.show()  
# COMPUTE OVERALL OOF AUC

oof = np.concatenate(oof_pred); true = np.concatenate(oof_tar);

names = np.concatenate(oof_names); folds = np.concatenate(oof_folds)

auc = roc_auc_score(true,oof)

print('Overall OOF AUC with TTA = %.3f'%auc)



# SAVE OOF TO DISK

df_oof = pd.DataFrame(dict(

    image_name = names, target=true, pred = oof, fold=folds))

df_oof.to_csv('oof.csv',index=False)

df_oof.head()
ds = get_dataset(files_test, augment=False, repeat=False, dim=IMG_SIZES[fold],

                 labeled=False, return_image_names=True)



image_names = np.array([img_name.numpy().decode("utf-8") 

                        for img, img_name in iter(ds.unbatch())])
submission = pd.DataFrame(dict(image_name=image_names, target=preds[:,0]))

submission = submission.sort_values('image_name') 

submission.to_csv('submission.csv', index=False)

submission.head()
plt.hist(submission.target,bins=100)

plt.show()