import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
"""Extract images from Avito's advertisement image zip archive.

Code adapted from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import zipfile

NUM_IMAGES_TO_EXTRACT = 10

with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])

!mv *.jpg/data/competition_files/train_jpg/* ~/avito_images
!rm -rf *.jpg
import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')
from PIL import Image
import cv2

def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]


def classify_and_plot(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    cv_img = cv2.imread(image_path)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    return (img, cv_img, preds_arr)
image_files = [x.path for x in os.scandir(images_dir)]
from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

def get_data_from_image(dat):
    plt.imshow(dat[0])
    img_size = [dat[0].size[0], dat[0].size[1]]
    (means, stds) = cv2.meanStdDev(dat[1])
    mean_color = np.mean(dat[1].flatten())
    std_color = np.std(dat[1].flatten())
    color_stats = np.concatenate([means, stds]).flatten()
    scores = [i[1][0][2] for i in dat[2]]
    labels = [i[1][0][1] for i in dat[2]]
    df = pd.DataFrame([img_size + [mean_color] + [std_color] + color_stats.tolist() + scores + labels],
                      columns = ['img_size_x', 'img_size_y', 'img_mean_color', 'img_std_color', 'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'image_green_std', 'image_red_std', 'Resnet50_score', 'xception_score', 'Inception_score', 'Resnet50_label', 'xception_label', 'Inception_label'])
    return df

dat = classify_and_plot(image_files[0])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[1])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[2])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[3])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[4])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[5])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[6])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[7])
df = get_data_from_image(dat)
print(df.head())
%%time
dat = classify_and_plot(image_files[8])
df = get_data_from_image(dat)
print(df.head())