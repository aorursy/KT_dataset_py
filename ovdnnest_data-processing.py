# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

import csv

import shutil

import random

import numpy as np

from PIL import Image

import pandas as pd

import cv2 as cv



if 1:

    for dirname, _, filenames in os.walk('/kaggle/input'):

        print(dirname)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def white_balance(img):

    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    avg_a = np.average(result[:, :, 1])

    avg_b = np.average(result[:, :, 2])

    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)

    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)

    return result
def resize_img_to_folder(img_name,diagnosis):

    image_loc = os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train',img_name+'.jpg')

    image = cv.imread(image_loc)

    new_img = cv.resize(image,(224,224))

    new_img = white_balance(new_img)

    dest = os.path.join('/kaggle/working/siim-isic-melanoma-classification/train_resized_downsampled',diagnosis)

    if not os.path.exists(dest):

        os.makedirs(dest)

    new_img = Image.fromarray(new_img)

    new_img.save(os.path.join(dest,img_name+'.jpg'))
train_path = '/kaggle/input/siim-isic-melanoma-classification/train.csv'



with open(train_path) as csvfile:

    reader = csv.reader(csvfile)

    nb_unknowns = 0

    if not os.path.exists('/kaggle/working/siim-isic-melanoma-classification'):

        os.makedirs('/kaggle/working/siim-isic-melanoma-classification')

    with open('/kaggle/working/siim-isic-melanoma-classification/downsampled_train_resized.csv', 'w', newline='') as file:

        writer = csv.writer(file)

        for row in reader:

            diagnosis = row[5]

            if diagnosis == 'unknown':

                nb_unknowns += 1

                nb_unknowns %= 10

                if not nb_unknowns:

                    resize_img_to_folder(row[0],diagnosis)

                    writer.writerow(row)

            elif not diagnosis == 'diagnosis':

                resize_img_to_folder(row[0],diagnosis)

                writer.writerow(row)

            else:

                writer.writerow(row)

        file.close()

    csvfile.close()