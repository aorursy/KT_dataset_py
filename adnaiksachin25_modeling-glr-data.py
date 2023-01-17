"""Baseline kernel for "Google Landmarks Recognition Challenge 2020".



Generates `submission.csv` in Kaggle format. When the number of training images

indicates that the kernel is being run against the public dataset,

simply copies `sample_submission.csv` to allow for quickly starting reruns

on the private dataset. When in a rerun against the private dataset,

makes predictions via retrieval, using DELG TensorFlow SavedModels for global

and local feature extraction.



First, ranks all training images by embedding similarity to each test image.

Then, performs geometric-verification and re-ranking on the `NUM_TO_RERANK`

most similar training images. For a given test image, each class' score is

the sum of the scores of re-ranked training images, and the predicted

class is the one with the highest aggregate score.



NOTE: For speed, this uses `pydegensac` as its RANSAC implementation.

Since the module has no interface for setting random seeds, RANSAC results

and submission scores will vary slightly between reruns.

"""



import copy

import csv

import gc

import operator

import os

import pathlib

import shutil



import pydegensac

from scipy import spatial

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from glob import glob

import gc

from PIL import Image

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline

import matplotlib.pyplot as plt

from IPython.display import Image,display

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

import matplotlib.image as mpimg

import scipy.spatial.distance as dist

from sklearn.model_selection import train_test_split

from skimage.measure import compare_ssim

import os



# Dataset parameters:

INPUT_DIR = os.path.join('..', 'input')



DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')
NUM_PUBLIC_TRAIN_IMAGES = 1580470 
mainPath = '/kaggle/input/landmark-recognition-2020/train/'



tr = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')

sub = pd.read_csv('/kaggle/input/landmark-recognition-2020/sample_submission.csv')

tab = tr.landmark_id.value_counts()







all_img_paths = [y for x in os.walk(mainPath) for y in glob(os.path.join(x[0], '*.jpg'))]

ids = [y for x in os.walk(mainPath) for y in glob(os.path.join(x[0]))]

all_filenames = []

for filepath in all_img_paths:

    FileName = os.path.basename(filepath)

    all_filenames.append(FileName)

path_dict = dict(zip(all_filenames,all_img_paths))





df=pd.DataFrame()

df['fname'] = all_filenames

df['pname'] = all_img_paths

df['id'] = list(map(lambda x: x[:-4], df.fname))



train = pd.merge(tr, df, on = 'id')





mainPath = '/kaggle/input/landmark-recognition-2020/test/'

all_img_paths = [y for x in os.walk(mainPath) for y in glob(os.path.join(x[0], '*.jpg'))]

ids = [y for x in os.walk(mainPath) for y in glob(os.path.join(x[0]))]

all_filenames = []

for filepath in all_img_paths:

    FileName = os.path.basename(filepath)

    all_filenames.append(FileName)

path_dict = dict(zip(all_filenames,all_img_paths))



test=pd.DataFrame()

test['fname'] = all_filenames

test['pname'] = all_img_paths



test['id'] = list(map(lambda x: x[:-4], test.fname))





def imtocsv(data):

    from PIL import Image

    nrow = []

    ncol = []

    pix = []

    for j in data.pname:

        image = Image.open(j)

        nrow.append(image.size[0])

        ncol.append(image.size[1])

        pix.append(image.size[0]*image.size[1])

    out = {'nrow': nrow, 'ncol': ncol, 'pix':pix}

    df= pd.DataFrame(out)

    df.insert(0,'id',data.id,True)

    return df



out=imtocsv(train)

train_50 = pd.merge(train, out, on = 'id')



del out

out=imtocsv(test)

test_50 = pd.merge(test, out, on = 'id')



tr_2 = train_50.copy()

ts_2 = test_50.copy()







#os.listdir('../input/siim-isic-melanoma-classification/jpeg/train/')

def imtocsv02(data, resize):

    from PIL import Image

    r=resize

    D=np.zeros((data.shape[0],2*r))

    k = 0

    for j in data.pname:

        image = Image.open(j)

        out=image.resize((r,r))

        out=np.array(out)

        out = np.resize(out, (r,r))

        #print(out.size)

        a1 =np.diag(out)

        a2 = np.diag(np.rot90(out))

        A =np.append(a1,a2)

        D[k,]=A

        k = k+1

    col_list = ['x' + str(x) for x in range(0,2*r)]

    df= pd.DataFrame(D,columns=col_list)

    

    df.insert(0,'id',data.id,True)

    return df



out=imtocsv02(tr_2,28)

train_28 = pd.merge(tr_2, out, on = 'id')



del out

out=imtocsv02(ts_2,28)

test_28 = pd.merge(ts_2, out, on = 'id')



train = train_28.copy()

test = test_28.copy()



del train_50

del test_50

del train_28

del test_28

del tr_2

del ts_2



y_tr = train['landmark_id'].copy()

tr = train.copy()

ts = test.copy()

image_id = ts.id.copy()

tr.drop(['id','landmark_id','fname','pname'],axis = 1, inplace = True)

ts.drop(['id','fname','pname'],axis = 1, inplace = True)





from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3,p=1)



knn.fit(tr,y_tr)

label = knn.predict(ts)

score = knn.predict_proba(ts)
def load_labelmap():

    with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:

        csv_reader = csv.DictReader(csv_file)

        labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

    return labelmap
def save_submission_csv(predictions=None):

    """Saves optional `predictions` as submission.csv.



    The csv has columns {id, landmarks}. The landmarks column is a string

  containing the label and score for the id, separated by a ws delimeter.



  If `predictions` is `None` (default), submission.csv is copied from

  sample_submission.csv in `IMAGE_DIR`.



  Args:

    predictions: Optional dict of image ids to dicts with keys {class, score}.

  """



    if predictions is None:

       # Dummy submission!

        shutil.copyfile(os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')

        return



    with open('submission.csv', 'w') as submission_csv:

        csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])

        csv_writer.writeheader()

        csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})
def main():

    labelmap = load_labelmap()

    num_training_images = len(labelmap.keys())

    print(f'Found {num_training_images} training images.')



    if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:

        print(f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission.')

        save_submission_csv()

        return



    _, post_verification_predictions = get_predictions(labelmap)

    save_submission_csv(post_verification_predictions)



main()