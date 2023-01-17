import cv2

import tqdm

import random

import numpy as np

import pandas as pd



import os

print(os.listdir('/kaggle/input'))
def get_test_image_ids(filename, sep):

    test_pd = pd.read_csv(filename, sep=sep, engine='python')

    test_image_ids = []



    for i, row in test_pd.iterrows():

        image_id = row[0]



        if image_id not in test_image_ids:

            test_image_ids.append(image_id)



    return test_image_ids



image_ids = get_test_image_ids('/kaggle/input/severstal-steel-defect-detection/sample_submission.csv', sep='[_,]')

num_samples = len(image_ids)

print('All samples: {}'.format(num_samples))
def mobilenetv2_gen(width, height, batch_size, image_ids, dirname):

    index = 0



    while True:

        x = np.zeros((batch_size, height, width, 3), dtype=np.float)



        for i in range(batch_size):

            image_id = image_ids[index]

            image = cv2.imread(dirname.format(image_id)) / 255.

            x[i] = cv2.resize(image, (width, height))

            index += 1



            if index == len(image_ids):

                index = 0



        yield x



def unet_gen(width, height, batch_size, image_ids, dirname):

    index = 0



    while True:

        x = np.zeros((batch_size, height, width, 1), dtype=np.float)



        for i in range(batch_size):

            image_id = image_ids[index]

            image = cv2.imread(dirname.format(image_id), cv2.IMREAD_GRAYSCALE) / 255.

            x[i] = cv2.resize(image, (width, height))[..., np.newaxis]

            index += 1



            if index == len(image_ids):

                index = 0



        yield x
from keras import backend as K



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
from itertools import groupby



def compress(image, height):

    inds = []



    for x in range(image.shape[1]):

        y = 0

        for val, grouper in groupby(image[:,x]):

            count = len(list(grouper))

            if val == True:

                inds.append(y + x * height)

                inds.append(count)

            y += count



    return inds



def get_resized_mask(preds, width, height, class_id):

    return cv2.resize(preds[:, :, class_id], (width, height)).round().astype(np.uint8) 
from keras.models import load_model



image_width = 1600

image_height = 256



width_mobile = 224

height_mobile = 224



width_unet = 800

height_unet = 128



batch_size = 32

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]



dirname = '/kaggle/input/severstal-steel-defect-detection/test_images/{}'



mobilenetv2 = load_model('/kaggle/input/severstal-mobilenetv2-classifier/classifier.h5')

unet_model = load_model('/kaggle/input/severstal-unet-model-no-resize/unet_model.h5', custom_objects={'dice_coef': dice_coef})

preds_rle = []



all_steps = int(np.ceil(num_samples / batch_size))

for i in tqdm.tqdm(range(all_steps)):

    image_ids_now = image_ids[i * batch_size: (i + 1) * batch_size]

    preds_mobile = mobilenetv2.predict_generator(mobilenetv2_gen(

        width=width_mobile, 

        height=height_mobile, 

        batch_size=batch_size, 

        image_ids=image_ids_now, 

        dirname=dirname), steps=1, verbose=0)

    image_ids_with_defect = [image_ids_now[index] for index in range(len(image_ids_now)) if preds_mobile[index] > 0.5]



    preds_unet = unet_model.predict_generator(unet_gen(

        width=width_unet, 

        height=height_unet, 

        batch_size=batch_size, 

        image_ids=image_ids_with_defect, 

        dirname=dirname), steps=np.ceil(len(image_ids_with_defect) / batch_size), verbose=0)



    preds = np.zeros((batch_size, height_unet, width_unet, 4), dtype=np.float)

    for i, image_id in enumerate(image_ids_with_defect):

        index = image_ids_now.index(image_id)

        preds[index] = preds_unet[i]



    for j in range(batch_size):

        image_id = image_ids[j]



        for class_id in range(len(colors)):

            image_mask = get_resized_mask(preds[j], image_width, image_height, class_id)

            indexes_rle = ''.join(str(x) + ' ' for x in compress(image_mask == 1, image_height))[:-1]

            preds_rle.append(indexes_rle)
preds_rle = preds_rle[:num_samples * 4]

df = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/sample_submission.csv', sep=',', engine='python')

df['EncodedPixels'] = preds_rle

df[['ImageId_ClassId', 'EncodedPixels']].to_csv('/kaggle/working/submission.csv', index=False)

df.head(n=32)