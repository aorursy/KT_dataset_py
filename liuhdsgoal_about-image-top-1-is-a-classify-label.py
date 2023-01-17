"""Copy Keras pre-trained model files to work directory from:
https://www.kaggle.com/gaborfodor/keras-pretrained-models

Code from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
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
import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

%matplotlib inline
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = InceptionResNetV2(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')
train_df = pd.read_csv('../input/avito-demand-prediction/train.csv',usecols=['image_top_1','item_id','image','category_name','parent_category_name'],index_col='item_id')
groups = train_df.groupby(['image_top_1',])
g0 = groups.get_group(0)
g1 = groups.get_group(1)
uc = g0['category_name'].unique()
uc
from PIL import Image
from zipfile import ZipFile
zip_path = '../input/avito-demand-prediction/train_jpg.zip'
# with ZipFile(zip_path) as myzip:
#     files_in_zip = myzip.namelist()
# with ZipFile(zip_path) as myzip:
#     with myzip.open(files_in_zip[3]) as myfile:
#         img = Image.open(myfile)
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


def plot_preds(img, preds_arr):
    """Plot image and its prediction."""
    sns.set_color_codes('pastel')
    f, axarr = plt.subplots(1, len(preds_arr) + 1, figsize=(20, 5))
    axarr[0].imshow(img)
    axarr[0].axis('off')
    for i in range(len(preds_arr)):
        _, x_label, y_label = zip(*(preds_arr[i][1]))
        plt.subplot(1, len(preds_arr) + 1, i + 2)
        ax = sns.barplot(x=y_label, y=x_label)
        plt.xlim(0, 1)
        ax.set()
        plt.xlabel(preds_arr[i][0])
    plt.show()


def classify_and_plot(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    plot_preds(img, preds_arr)
def classify_and_plot0(img):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    plot_preds(img, preds_arr)
myzip = ZipFile(zip_path)
def classify_and_plot_path(img_path):
    with myzip.open(img_path) as myfile:
        img = Image.open(myfile)
        classify_and_plot0(img)
import math
def plot_images(imgs):
    """Plot image and its prediction."""
    sns.set_color_codes('pastel')
    f, axarr = plt.subplots(math.ceil(len(imgs)/4.0), 4, figsize=(20, 5*(len(imgs)//4+1)))
    axarr= axarr.reshape((-1))
    for i,img in enumerate(imgs):
        axarr[i].imshow(img)
        axarr[i].axis('off')

    plt.show()
def plot_images_ids(paths):
    imgs = []
    for p in paths:
        myfile = myzip.open('data/competition_files/train_jpg/{}.jpg'.format(p))
        img = Image.open(myfile)
        imgs.append(img)
    plot_images(imgs)
plot_images_ids(g0['image'][:40])

for i in range(0,100,5):
    p = 'data/competition_files/train_jpg/{}.jpg'.format(g0['image'][i])
    classify_and_plot_path(p)
plot_images_ids(g1['image'][:40])
for i in range(0,100,5):
    p = 'data/competition_files/train_jpg/{}.jpg'.format(g1['image'][i])
    classify_and_plot_path(p)