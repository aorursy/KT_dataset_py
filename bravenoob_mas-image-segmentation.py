#!pip install pyforest

#active_imports()
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
DATA_DIR = '../input/petfinder-adoption-prediction/'



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

    

    return round(cat_area_percent, 2), round(dog_area_percent, 2)
def mark_gray_image(df, image_path): 

    head, filename = os.path.split(image_path)

    new_row = pd.Series({"imageId": filename , "gray_image": 1, "cats":0, "dogs": 0, "cat_percent": 0, "dog_percent": 0})

    return df.append(new_row, ignore_index=True)
# Directory of images to run detection on

IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
results = model.detect([skimage.io.imread('../input/petfinder-adoption-prediction/train_images/e6bc32c73-1.jpg')], verbose=0)

results[0]
pet_area_coverage=pd.DataFrame()

failed_images = []

counter = 0

for filename in os.listdir(IMAGE_DIR):

    try:

        image_number = filename[filename.find('-')+1 :-4]

        if(image_number != '1'):

            continue   

        counter = counter +1

        if counter % 1000 == 0:

            print(counter)

        pet_area_coverage = calculate_areas(pet_area_coverage,os.path.join(IMAGE_DIR, filename))

    except Exception:

        failed_images.append(filename)

        pet_area_coverage = mark_gray_image(pet_area_coverage,os.path.join(IMAGE_DIR, filename))

        continue
for img in failed_images:

    print(img)
pet_area_coverage.to_csv('pet_area_coverage.csv',index=False)
pet_area_coverage.describe()
#show images with most cats and dogs in them



max_dog_count_petid = pet_area_coverage.loc[pet_area_coverage['dogs'].idxmax()]['imageId']

max_cat_count_petid = pet_area_coverage.loc[pet_area_coverage['cats'].idxmax()]['imageId']



calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_dog_count_petid), show_image=True)

calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_cat_count_petid), show_image=True)
# show images with most area covered by cats and dogs



max_dog_percent_petid = pet_area_coverage.loc[pet_area_coverage['dog_percent'].idxmax()]['imageId']

max_cat_percent_petid = pet_area_coverage.loc[pet_area_coverage['cat_percent'].idxmax()]['imageId']



calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_dog_percent_petid), show_image=True)

calculate_areas(pd.DataFrame(),os.path.join(IMAGE_DIR, max_cat_percent_petid), show_image=True)
train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
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

    for i, row in matching_rows.iterrows():

        img_nr = row['imageIndex']

        gray_images = gray_images + row['gray_image']

        pet_photo_values = pet_photo_values.append(pd.Series([row['cat_percent'], row['cats'], row['dog_percent'], row['dogs']],

                                         index = ['cat_percent_'+img_nr, 'cats_'+img_nr, 'dog_percent_'+img_nr, 'dogs_'+img_nr] ))



    pet_photo_values = pet_photo_values.append(pd.Series([gray_images],index = ['gray_images'] ))

        

    # return series of all values   

    return pet_photo_values
result_df=pd.DataFrame()

result_df['PetID']=train_df['PetID']

result_df['AdoptionSpeed']=train_df['AdoptionSpeed']

result_df = result_df.merge(result_df.PetID.apply(lambda x: add_image_segmentations(pet_area_coverage, x)), left_index=True, right_index=True)
result_df.fillna(0, inplace=True)
result_df.shape
#Machine Learning Algorithm (MLA) Selection and Initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #xgboost: 

    XGBClassifier(),

    

    #CatBoostClassifier(verbose=0)

    ]

def train_model(data, MLA_list = MLA):

    

    target = data['AdoptionSpeed']

    X_train = data.drop(['AdoptionSpeed'],axis=1)

    

    MLA_columns = ['MLA Name', 'MLA Parameters','MLA cohen_kappa_score','MLA Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)



    MLA_predict = data['AdoptionSpeed']

    

    row_index = 0

    for alg in MLA_list:



        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

        kf = StratifiedKFold(n_splits=10, shuffle=True)

        kappa_score = make_scorer(cohen_kappa_score, weights='quadratic')

        cv_results = model_selection.cross_validate(alg, X_train, target, cv  = kf, scoring=kappa_score )

        

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

        MLA_compare.loc[row_index, 'MLA cohen_kappa_score'] = cv_results['test_score'].mean() 

        MLA_compare.loc[row_index, 'algo'] = alg.__class__

             

        #MLA_predict[MLA_name] = alg.predict(X_train)

        row_index+=1



    MLA_compare.sort_values(by = ['MLA cohen_kappa_score'], ascending = False, inplace = True)

    sns.barplot(x='MLA cohen_kappa_score', y = 'MLA Name', data = MLA_compare, color = 'b')

    plt.title('Machine Learning Algorithm Accuracy Score \n')

    plt.xlabel('Accuracy Score (%)')

    plt.ylabel('Algorithm')

    

    return MLA_compare
classifier_performance = train_model(result_df.drop(['PetID'], axis=1))
classifier_performance
X_test = result_df.drop(['PetID', 'AdoptionSpeed'], axis=1)

target = result_df['AdoptionSpeed']



best_classifier = classifier_performance.iloc[0][4]



classifier = best_classifier().fit(X_test, target)
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test_images")

test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
test_pet_area_coverage=pd.DataFrame()

counter = 0

for filename in os.listdir(TEST_IMAGE_DIR):

    try:

        image_number = filename[filename.find('-')+1 :-4]

        if(image_number != '1'):

            continue   

        counter = counter +1

        if counter % 1000 == 0:

            print(counter)

        test_pet_area_coverage = calculate_areas(test_pet_area_coverage,os.path.join(TEST_IMAGE_DIR, filename))

    except Exception:

        test_pet_area_coverage = mark_gray_image(test_pet_area_coverage,os.path.join(TEST_IMAGE_DIR, filename))

        continue
test_result_df=pd.DataFrame()

test_result_df['PetID']=test_df['PetID']

test_result_df = test_result_df.merge(test_result_df.PetID.apply(lambda x: add_image_segmentations(test_pet_area_coverage, x)), left_index=True, right_index=True)
test_result_df.fillna(0, inplace=True)
test_result_df.shape
submit=pd.DataFrame()

submit['PetID']=test_result_df['PetID']

submit['AdoptionSpeed']=classifier.predict(test_result_df.drop(['PetID'], axis=1))

submit['AdoptionSpeed']=submit['AdoptionSpeed'].astype(int)

submit.to_csv('submission.csv',index=False)
import shutil

shutil.rmtree(MASK_DIR)