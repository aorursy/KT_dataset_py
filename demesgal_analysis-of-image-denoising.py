import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from skimage import io

import cv2

import matplotlib.pyplot as plt
# !git clone https://github.com/yzhouas/PD-Denoising-pytorch.git

# !mv PD-Denoising-pytorch denoise

!mkdir data

!mkdir data/img
import shutil

from glob import glob

FOLDERS = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']

SRC_FOLDER = '../input/alaska2-image-steganalysis'

FILTERED_FOLDER = '../input/alaska2filtered-images/filtered'

FILTERED_POSTFIX = '_pss2_k0.0.png'

JPEG_POSTFIX = '.jpg'

dataset = []

for path in glob(f'{FILTERED_FOLDER}/*{FILTERED_POSTFIX}'):

    new_img = path.split('/')[-1].replace(FILTERED_POSTFIX, '').split('_')[-1]

    if new_img not in dataset:

        dataset.append(new_img)

    
def read_img(path):

    """ Reads image. Image is a float version of uint8 format 0..255"""

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    return image

    

def calc_diff_features(src_path, filterd_path):

    """ Calculate the difference between images. """

    img = read_img(src_path)

    filtered_img = read_img(filterd_path)

    diff_img = (filtered_img - img) / filtered_img

    hist = np.histogram(diff_img, bins=24, range=(0.8, 1.3))

    

    return hist

    

def plot_diff(path1, path2):

    """ Plots the difference between images. """

    image = read_img(path1)

    base_image = read_img(path2)

    abs_diff = np.abs(image - base_image)

    print(f'{path1.split("/")[-1]} and {path2.split("/")[-1]}')

    print(f'MAE: {np.mean(abs_diff)}')

    

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    ax.set_axis_off()

    ax.imshow(0.5 + (image - base_image) / np.max(abs_diff));
datatable = []

labels = []

for img_id in dataset:

    for (label, folder) in enumerate(FOLDERS):

        p = calc_diff_features(f'{SRC_FOLDER}/{folder}/{img_id}{JPEG_POSTFIX}', f'{FILTERED_FOLDER}/{folder}_{img_id}{FILTERED_POSTFIX}')

        datatable.append(p[0])

        labels.append(label)
datatable = np.array(datatable)

labels = np.array(labels)
datatable.shape
train_X = datatable[0:700]

test_X = datatable[700:900]

train_Y = labels[0:700]

test_Y = labels[700:900]

train_X
from sklearn import metrics



tpr_thresholds = [0.0, 0.4, 1.0]

weights = [2, 1]

areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

normalization = np.dot(areas, weights)

print(normalization)

    

def alaska_weighted_auc(y_true, y_valid):

    """

    https://www.kaggle.com/anokas/weighted-auc-metric-updated

    """

    

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid)



    competition_metric = 0

    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)

        # pdb.set_trace()



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min  # normalize such that curve starts at y=0

        score = metrics.auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric



    return competition_metric / normalization
import xgboost as xgb
gbm1 = xgb.XGBClassifier(max_depth=5, n_estimators=750, learning_rate=0.05, random_state=0).fit(train_X, train_Y > 0)

predictions = gbm1.predict_proba(test_X)

print(f'ROC-AUC: {alaska_weighted_auc(test_Y > 0, predictions[:,1])}')
gbm2 = xgb.XGBClassifier(max_depth=5, n_estimators=750, learning_rate=0.05, random_state=0).fit(train_X, train_Y)

predictions = gbm2.predict_proba(test_X)
# Set the encrypted class value to the greatest one

predictions = np.transpose(np.vstack((predictions[:, 0], np.max(predictions[:, 1:3], axis = 1))))

print(f'ROC-AUC: {alaska_weighted_auc(test_Y > 0, predictions[:,1])}')
# Normalize

predictions = predictions / np.reshape(np.sum(predictions, axis = 1),(predictions.shape[0],1))

print(f'ROC-AUC: {alaska_weighted_auc(test_Y > 0, predictions[:,1])}')