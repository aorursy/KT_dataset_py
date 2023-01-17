import os

import sys

import numpy as np

import pandas as pd

import keras

from matplotlib import pyplot as plt

from skimage.io import imread, imshow 

from skimage.transform import resize

from skimage.morphology import label

from PIL import Image, ImageDraw

from tqdm.notebook import tqdm

from skimage.measure import label, regionprops



sys.path.insert(0, '/kaggle/input/efficientnet-keras-source-code/')

import efficientnet.keras
model = keras.models.load_model('/kaggle/input/gwd-starter-gpu-v2-train/model.h5', compile=False)
IMG_WIDTH = 256

IMG_HEIGHT = 256

IMG_CHANNELS = 3

PATH = "../input/global-wheat-detection/"

TEST_PATH = '/kaggle/input/global-wheat-detection/test/'

test_folder = os.path.join(PATH, "test")

sample_sub = pd.read_csv(PATH + "sample_submission.csv")

test_ids = os.listdir(TEST_PATH)
# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path)[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img
%%time



THRESH = 0.75



preds = model.predict(X_test)[:, :, :, 0]

masked_preds = preds > THRESH
n_rows = 3



f, ax = plt.subplots(n_rows, 3, figsize=(14, 10))



for j, idx in enumerate([4,5,6]):

    for k, kind in enumerate(['original', 'pred', 'masked_pred']):

        if kind == 'original':

            img = X_test[idx]

        elif kind == 'pred':

            img = preds[idx]

        elif kind == 'masked_pred':

            masked_pred = preds[idx] > .75

            img = masked_pred

        ax[j, k].imshow(img)



plt.tight_layout()
def get_params_from_bbox(coords, scaling_factor=1):

    xmin, ymin = coords[1] * scaling_factor, coords[0] * scaling_factor

    w = (coords[3] - coords[1]) * scaling_factor

    h = (coords[2] - coords[0]) * scaling_factor

    

    return xmin, ymin, w, h
# Allows to extract bounding boxes from binary masks

bboxes = list()



for j in range(masked_preds.shape[0]):

    label_j = label(masked_preds[j, :, :]) 

    props = regionprops(label_j)

    bboxes.append(props)
output = dict()

for i in range(masked_preds.shape[0]):

    bboxes_processed = [get_params_from_bbox(bb.bbox, scaling_factor=4) for bb in bboxes[i]]

    formated_boxes = ['1.0 ' + ' '.join(map(str, bb_m)) for bb_m in bboxes_processed]

    

    output[sample_sub["image_id"][i]] = " ".join(formated_boxes)
sample_sub["PredictionString"] = output.values()
sample_sub.head()
sample_sub.to_csv('submission.csv', index=False)