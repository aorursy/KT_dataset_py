# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder, StandardScaler,KBinsDiscretizer

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
%reload_ext autoreload

%autoreload 2

%matplotlib inline
DATA_DIR = '/kaggle/input/'



# Directory to save logs and trained model

ROOT_DIR = '/kaggle/working/'



MASK_DIR = '/kaggle/working/Mask_RCNN'
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')

!python setup.py -q install

os.chdir('..')
!pip show mask-rcnn
import os

import sys

import random

import math

import numpy as np

import skimage.io

import matplotlib

import matplotlib.pyplot as plt



sys.path.append(MASK_DIR) # To find local version

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.config import Config



# Directory to save logs and trained model

MODEL_DIR = os.path.join(ROOT_DIR, "logs")



# Local path to trained weights file

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed

if not os.path.exists(COCO_MODEL_PATH):

    utils.download_trained_weights(COCO_MODEL_PATH)



# Directory of images to run detection on

IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
class InferenceConfig(Config):

    # Set batch size to 1 since we'll be running inference on

    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NAME = "mas_segmentation"

    NUM_CLASSES = 1 + 80



config = InferenceConfig()
# Create model object in inference mode.

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)



# Load weights trained on MS-COCO

model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names

# Index of the class in the list is its ID. For example, to get ID of

# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',

               'bus', 'train', 'truck', 'boat', 'traffic light',

               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',

               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',

               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',

               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',

               'kite', 'baseball bat', 'baseball glove', 'skateboard',

               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',

               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',

               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',

               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',

               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',

               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',

               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',

               'teddy bear', 'hair drier', 'toothbrush']
def calculate_areas(df, image_path, show_image=False): 

    head, filename = os.path.split(image_path)

    image = skimage.io.imread(image_path)



    # Run detection

    results = model.detect([image], verbose=0)

    r = results[0]

    

    if show_image:

        # Visualize results

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])



    number_of_cats = int((r['class_ids'] == 16).sum())

    number_of_dogs = int((r['class_ids'] == 17).sum())

    

    cat_area_percent, dog_area_percent = calculate_area_precent(r)

    

    new_row = pd.Series({"imageId": filename , "gray_image": 0, "cats": number_of_cats, "dogs": number_of_dogs, "cat_percent": cat_area_percent, "dog_percent": dog_area_percent})

    return df.append(new_row, ignore_index=True)
def calculate_area_precent(result):



    #mask as (x,y, number_of_objects)

    mask = result['masks']

    mask = mask.astype(int)

    

    cat_mask_index = np.where(result['class_ids'] == 16)[0]

    dog_mask_index = np.where(result['class_ids'] == 17)[0]

        

    cat_area = 0

    dog_area = 0

    for i in cat_mask_index:

        cat_area = cat_area + np.sum(mask[:,:,i])



    for i in dog_mask_index:

        dog_area = dog_area + np.sum(mask[:,:,i])



    mask_size = mask.shape[0] * mask.shape[1]   



    cat_area_percent = cat_area/mask_size

    dog_area_percent = dog_area/mask_size

    

    return round(cat_area_percent,2), round(dog_area_percent,2)
def mark_gray_image(df, image_path): 

    head, filename = os.path.split(image_path)

    new_row = pd.Series({"imageId": filename , "gray_image": 1, "cats":0, "dogs": 0, "cat_percent": 0, "dog_percent": 0})

    return df.append(new_row, ignore_index=True)
# data load from previous commit.

#pet_area_coverage=pd.DataFrame()

#failed_images = []

#counter = 0

#for filename in os.listdir(IMAGE_DIR):

#    try:

#        counter = counter +1

#        if counter % 1000 == 0:

#            print(counter)

#        pet_area_coverage = calculate_areas(pet_area_coverage,os.path.join(IMAGE_DIR, filename))

#    except Exception:

#        failed_images.append(filename)

#        pet_area_coverage = mark_gray_image(pet_area_coverage,os.path.join(IMAGE_DIR, filename))

#        continue
#for img in failed_images:

#    print(img)
#pet_area_coverage.to_csv('pet_area_coverage.csv',index=False)
#pet_area_coverage.describe()
#show images with most cats and dogs in them



#max_dog_count_petid = pet_area_coverage.loc[pet_area_coverage['dogs'].idxmax()]['imageId']

#max_cat_count_petid = pet_area_coverage.loc[pet_area_coverage['cats'].idxmax()]['imageId']



#calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_dog_count_petid), show_image=True)

#calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_cat_count_petid), show_image=True)
# show images with most area covered by cats and dogs



#max_dog_percent_petid = pet_area_coverage.loc[pet_area_coverage['dog_percent'].idxmax()]['imageId']

#max_cat_percent_petid = pet_area_coverage.loc[pet_area_coverage['cat_percent'].idxmax()]['imageId']



#calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_dog_percent_petid), show_image=True)

#calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_cat_percent_petid), show_image=True)
train_df = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/train/train.csv")



# load from dataset

pet_area_coverage = pd.read_csv("/kaggle/input/rcnn-all-train-images-results/pet_area_coverage.csv")
def add_image_segmentations(df, PetID):

    #df = area cover df with imageId

    matching_rows = df[df['imageId'].str.contains(PetID)]

    count_row = matching_rows.shape[0]

        

    if count_row == 0:

        # no images for this pet

        return pd.Series()

    

    matching_rows['imageIndex'] = matching_rows.apply(lambda x: x['imageId'][x['imageId'].find('-')+1 :-4], axis=1)

    matching_rows = matching_rows.sort_values('imageIndex')

        

    # add values from all images

    pet_photo_values = pd.Series()

    gray_images = 0

    cat_count_max = 0

    dog_count_max = 0

    cat_count_mean = 0

    dog_count_mean = 0

    cat_percent_max = 0

    dog_percent_max = 0

    cat_percent_mean = 0

    dog_percent_mean = 0

    

    for i, row in matching_rows.iterrows():

        img_nr = row['imageIndex']

        gray_images = gray_images + row['gray_image']

        

        if row['cats'] > cat_count_max:

            cat_count_max = row['cats']

        if row['dogs'] > dog_count_max:

            dog_count_max = row['dogs']

        

        cat_count_mean = (cat_count_mean + row['cats'])/2

        dog_count_mean = (dog_count_mean + row['dogs'])/2

            

        if row['cat_percent'] > cat_percent_max:

            cat_percent_max = row['cat_percent']

        if row['dog_percent'] > dog_percent_max:

            dog_percent_max = row['dog_percent']

        

        cat_percent_mean = (cat_percent_mean + row['cat_percent'])/2

        dog_percent_mean = (dog_percent_mean + row['dog_percent'])/2

        

    pet_photo_values = pet_photo_values.append(pd.Series([gray_images, cat_count_max, dog_count_max, cat_count_mean, dog_count_mean, cat_percent_max,

                                                          dog_percent_max, cat_percent_mean, dog_percent_mean ],index = ['gray_images', 'cat_count_max', 

                                                                                                                         'dog_count_max', 'cat_count_mean', 

                                                                                                                         'dog_count_mean', 'cat_percent_max',

                                                                                                                         'dog_percent_max', 'cat_percent_mean', 

                                                                                                                         'dog_percent_mean'] ))     

    # return series of all values   

    return pet_photo_values
result_df=pd.DataFrame()

result_df['PetID']=train_df['PetID']

result_df['AdoptionSpeed']=train_df['AdoptionSpeed']

result_df = result_df.merge(result_df.PetID.apply(lambda x: add_image_segmentations(pet_area_coverage, x)), left_index=True, right_index=True)
result_df.sample(10)
result_df.fillna(0, inplace=True)
result_df.shape
result_df.to_csv('result.csv',index=False)
import shutil

shutil.rmtree(MASK_DIR)