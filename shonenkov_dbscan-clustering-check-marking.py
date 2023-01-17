from glob import glob

import json

from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import GroupKFold

from sklearn.cluster import DBSCAN

from collections import defaultdict

import cv2

from skimage import io

import albumentations as A

import scipy as sp

import torch

import os

from datetime import datetime

import time

import random

import cv2

import pandas as pd

import numpy as np

import albumentations as A

import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torch.nn import functional as F

from glob import glob

import sklearn

from torch import nn

import warnings



warnings.filterwarnings("ignore") 

warnings.filterwarnings("ignore", category=DeprecationWarning) 



SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
DATA_PATH = '../input/melanoma-merged-external-data-512x512-jpeg'



df_folds = pd.read_csv(f'{DATA_PATH}/folds_08062020.csv', index_col='image_id')



TRAIN_IMAGE_IDS = df_folds.index.values

TRAIN_CLUSTERS = []

TRAIN_CLUSTERING_CACHE = set()

BAD_CASES_CACHE = set()
!pip install --no-deps imagededup==0.2.2 > /dev/null
%%time



from imagededup.methods import PHash



phasher = PHash()



encodings = phasher.encode_images(image_dir=f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma')

duplicates = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=0)
# Not duplicates but found as duplicates using imagededup (manually checked):

BAD_CASES = [

    ['ISIC_0148783', 'ISIC_2016546'],

    ['ISIC_2697895', 'ISIC_4591526'],

    ['ISIC_6927689', 'ISIC_7106012'],

    ['ISIC_6088194', 'ISIC_7559729'],

    ['ISIC_3840133', 'ISIC_7168301'],

    ['ISIC_0641480', 'ISIC_3273370'],

    ['ISIC_4007215', 'ISIC_7219333'],

    ['ISIC_2372032', 'ISIC_2484560'],

    ['ISIC_3107526', 'ISIC_3900903'],

    ['ISIC_3828455', 'ISIC_6577087'],

    ['ISIC_1536014', 'ISIC_5709767'],

    ['ISIC_0033743', 'ISIC_1356715'],

    ['ISIC_5044766', 'ISIC_7912183'],

    ['ISIC_4089379', 'ISIC_8432001'],

    ['ISIC_0188415', 'ISIC_9626241'],

    ['ISIC_4848047', 'ISIC_9117456'],

    ['ISIC_0032336', 'ISIC_9125216'],

    ['ISIC_0030513', 'ISIC_0688622'],

    ['ISIC_3520750', 'ISIC_9639348'],

    ['ISIC_2496831', 'ISIC_9540109'],

    ['ISIC_1410153', 'ISIC_5950041'],

    ['ISIC_3866081', 'ISIC_6754247'],

    ['ISIC_2129226', 'ISIC_2647198'],

    ['ISIC_0148783', 'ISIC_2016546', 'ISIC_3455285', 'ISIC_7460560'],

    ['ISIC_2697895', 'ISIC_4591526', 'ISIC_6625344', 'ISIC_9367832'],

    ['ISIC_0789732', 'ISIC_8303710', 'ISIC_9167141'],

]



for case in BAD_CASES:

    BAD_CASES_CACHE.add('.'.join(sorted(case)))
for image_id, values in tqdm(duplicates.items(), total=len(duplicates)):

    image_id = image_id.split('.')[0]

    if len(values) < 1:

        continue

    if image_id not in TRAIN_IMAGE_IDS:

        continue

    sorted_cluster = [image_id]

    for value in values:

        value = value.split('.')[0]

        if value in TRAIN_IMAGE_IDS:

            sorted_cluster.append(value)



    sorted_cluster = sorted(sorted_cluster)

    if len(sorted_cluster) > 1:

        cluster_name = '.'.join(sorted_cluster)

        if cluster_name in BAD_CASES_CACHE:

            continue

        if cluster_name not in TRAIN_CLUSTERING_CACHE:

            TRAIN_CLUSTERING_CACHE.add(cluster_name)

            TRAIN_CLUSTERS.append(sorted_cluster)

            

TRAIN_CLUSTERS = sorted(TRAIN_CLUSTERS, key=lambda x: -len(x))
margin = 0

count = 20



draw_clusters = TRAIN_CLUSTERS[margin:margin+count]



size = min([5, len(draw_clusters[0])])



fig, ax = plt.subplots(count, size, figsize=(size*3, 4*count))



for j, image_ids in enumerate(draw_clusters):

    for i, image_id in enumerate(image_ids[:size]):

        image_id = image_id.split('.')[0]

        image = cv2.imread(f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        patient_id = df_folds.loc[image_id]['patient_id']

        ax[j][i].set_title(f'{patient_id}\n{image_id}')
print('-'*10 + '[imagededup]' + '-'*10)

print(f'[Clusters found]:', len(TRAIN_CLUSTERS))

print(f'[Precision]: ~{1 - round(len(BAD_CASES_CACHE) / (len(TRAIN_CLUSTERS) + len(BAD_CASES_CACHE)), 3)}')

print('-'*32)
TRAIN_DBSCAN_CLUSTERS = []

TRAIN_DBSCAN_CLUSTERING_CACHE = set()

BAD_CASES_DBSCAN_CACHE = set()
train_embeddings = np.load('../input/melanoma-image-embeddings/train_embeddings.npy')

train_image_names = json.load(open('../input/melanoma-image-embeddings/train_image_names.json', 'rb'))
%%time



clusters = defaultdict(list)

for image_name, cluster_id in zip(train_image_names, DBSCAN(eps=3.0, min_samples=1, n_jobs=4).fit_predict(train_embeddings)):

    clusters[cluster_id].append(image_name)
dbscan_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

sorted_dbscan_clusters = [

    image_ids

    for _, image_ids in dbscan_clusters if 1 < len(image_ids) <= 5

]
BAD_CASES_DBSCAN = [

    ['ISIC_0063674', 'ISIC_0073214'],

    ['ISIC_0025789', 'ISIC_0031713'],

    ['ISIC_0024371', 'ISIC_0064216'],

    ['ISIC_2697083', 'ISIC_5258657'],

    ['ISIC_1665944', 'ISIC_3475660'],

    ['ISIC_0959735', 'ISIC_2028658'],

    ['ISIC_0645454', 'ISIC_0851556'],

    ['ISIC_0268080', 'ISIC_7364244'],

    ['ISIC_0188432', 'ISIC_2459552'],

    ['ISIC_0058863', 'ISIC_0062880', 'ISIC_0068056', 'ISIC_0068631'],

    ['ISIC_1068686', 'ISIC_4214813', 'ISIC_5844037'],

    ['ISIC_3593913', 'ISIC_4569978', 'ISIC_8509430'],

]



for case in BAD_CASES_DBSCAN:

    BAD_CASES_DBSCAN_CACHE.add('.'.join(sorted(case)))
for image_ids in tqdm(sorted_dbscan_clusters, total=len(sorted_dbscan_clusters)):

    sorted_cluster = []

    for image_id in image_ids:

        sorted_cluster.append(image_id)



    sorted_cluster = sorted(sorted_cluster)

    if len(sorted_cluster) > 1:

        cluster_name = '.'.join(sorted_cluster)

        if cluster_name in BAD_CASES_DBSCAN_CACHE:

            continue

        if cluster_name not in TRAIN_DBSCAN_CLUSTERING_CACHE:

            TRAIN_DBSCAN_CLUSTERING_CACHE.add(cluster_name)

            TRAIN_DBSCAN_CLUSTERS.append(sorted_cluster)



TRAIN_DBSCAN_CLUSTERS = sorted(TRAIN_DBSCAN_CLUSTERS, key=lambda x: -len(x))
len(TRAIN_DBSCAN_CLUSTERS)
margin = 0

count = 20



draw_clusters = TRAIN_DBSCAN_CLUSTERS[margin:margin+count]



size = min([5, len(draw_clusters[0])])



fig, ax = plt.subplots(count, size, figsize=(size*3, 4*count))



for j, image_ids in enumerate(draw_clusters):

    for i, image_id in enumerate(image_ids[:size]):

        image_id = image_id.split('.')[0]

        image = cv2.imread(f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        patient_id = df_folds.loc[image_id]['patient_id']

        ax[j][i].set_title(f'{patient_id}\n{image_id}')
print('-'*10 + '[DBSCAN]' + '-'*10)

print(f'[Clusters found]:', len(TRAIN_DBSCAN_CLUSTERS))

print(f'[Precision]: ~{1 - round(len(BAD_CASES_DBSCAN_CACHE) / (len(TRAIN_DBSCAN_CLUSTERS) + len(BAD_CASES_DBSCAN_CACHE)), 3)}')

print('-'*32)
intersection = TRAIN_CLUSTERING_CACHE.intersection(TRAIN_DBSCAN_CLUSTERING_CACHE)



print(f'DBSCAN results contain {len(intersection)} cases from imagededup results ({len(TRAIN_CLUSTERING_CACHE)})')
def check_subcase(case):

    case_image_ids = case.split('.')

    for dbscan_case in TRAIN_DBSCAN_CLUSTERING_CACHE:

        dbscan_case_image_ids = dbscan_case.split('.')

        if len(set(dbscan_case_image_ids).intersection(case_image_ids)) == len(case_image_ids):

            return True

    return False
ORIGINALS_CLUSTERS = []

for case in list(TRAIN_CLUSTERING_CACHE.difference(intersection)):

    if check_subcase(case):

        continue

    ORIGINALS_CLUSTERS.append(case.split('.'))



len(ORIGINALS_CLUSTERS)
RESULT_CLUSTERS = sorted(ORIGINALS_CLUSTERS + TRAIN_DBSCAN_CLUSTERS, key=lambda x: -len(x))

print('-'*30)

print('CLUSTERS COUNT:', len(RESULT_CLUSTERS))

print('DUPLICATED IMAGES COUNT:', sum([len(image_ids) for image_ids in RESULT_CLUSTERS]) )

print('-'*30)
TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
data = []

for image_ids in RESULT_CLUSTERS:

    sample = {}

    sample['image_ids'] = '.'.join(image_ids)

    sample.update(df_folds.loc[image_ids][['patient_id', 'target', 'source', 'sex', 'age_approx', 'anatom_site_general_challenge']].nunique())

    data.append(sample)



data = pd.DataFrame(data)

image_ids = [image_id[0] for image_id in data['image_ids'].str.split('.')]
data.head()
print(df_folds.loc[image_ids]['target'].value_counts())

df_folds.loc[image_ids]['target'].hist();
print(df_folds.loc[image_ids]['source'].value_counts())

df_folds.loc[image_ids]['source'].hist();
print(df_folds.loc[image_ids]['sex'].value_counts())

df_folds.loc[image_ids]['sex'].hist();
df_folds.loc[image_ids]['age_approx'].hist(bins=50);
df_folds.loc[image_ids]['anatom_site_general_challenge'].value_counts()
source_diff = data[data['source'] != 1]

count = source_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*2*count))

for j, (_, row) in enumerate(source_diff.iterrows()):

    image_ids = row['image_ids'].split('.')

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{image_id}, source: {df_folds.loc[image_id].source}')
target_diff = data[data['target'] != 1]

count = target_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*2*count))

for j, (_, row) in enumerate(target_diff.iterrows()):

    image_ids = row['image_ids'].split('.')

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{image_id}, target: {df_folds.loc[image_id].target}')
sex_diff = data[data['sex'] != 1]

count = sex_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*2*count))

for j, (_, row) in enumerate(sex_diff.iterrows()):

    image_ids = row['image_ids'].split('.')

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{image_id}, sex: {df_folds.loc[image_id].sex}')
anatom_diff = data[data['anatom_site_general_challenge'] != 1]

count = anatom_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*2*count))

for j, (_, row) in enumerate(anatom_diff.iterrows()):

    image_ids = row['image_ids'].split('.')

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{image_id}, {df_folds.loc[image_id].anatom_site_general_challenge}')
patient_diff = data[data['patient_id'] != 1]

count = patient_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*3*count))

for j, (_, row) in enumerate(patient_diff.iterrows()):

    image_ids = row['image_ids'].split('.')[:2]

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{df_folds.loc[image_id].patient_id}\n{image_id}')
age_diff = data[data['age_approx'] != 1]

count = age_diff.shape[0]

count
fig, ax = plt.subplots(count, 2, figsize=(8, 2*2*count))

for j, (_, row) in enumerate(age_diff.iterrows()):

    image_ids = row['image_ids'].split('.')

    for i, image_id in enumerate(image_ids):

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

        ax[j][i].imshow(image);

        ax[j][i].set_title(f'{image_id}, {df_folds.loc[image_id].age_approx}')
marking_diff = data[

    (data['target'] != 1) |

    (data['sex'] != 1) |

    (data['anatom_site_general_challenge'] != 1) |

    (data['age_approx'] != 1)

]

marking_diff.shape[0]
data.to_csv('duplicates.csv', index=False)