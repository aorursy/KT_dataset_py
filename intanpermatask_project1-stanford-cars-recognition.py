%reload_ext autoreload

%autoreload 2

%matplotlib inline



import os

import pandas as pd

import numpy as np

from pathlib import Path

import json

from PIL import ImageDraw, ImageFont

import cv2

import shutil

import random

from matplotlib import patches, patheffects

import matplotlib.pyplot as plt



import scipy.io as sio



from fastai import *

from fastai.vision import *
def get_labels():

    annos = sio.loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_annos.mat')

    _, total_size = annos["annotations"].shape

    print("total sample size is ", total_size)

    labels = np.zeros((total_size, 5))

    for i in range(total_size):

        path = annos["annotations"][:,i][0][0][0].split(".")

        id = int(path[0][8:]) - 1

        for j in range(5):

            labels[id, j] = int(annos["annotations"][:,i][0][j + 1][0])

    return labels

labels = get_labels()

labels
devkit_path = Path('../input/stanford-cars-dataset/car_devkit/devkit')

train_path = Path('../input/stanford-cars-dataset/cars_train/cars_train')

test_path = Path('../input/stanford-cars-dataset/cars_test/cars_test')
os.listdir(devkit_path)
cars_meta = loadmat(devkit_path/'cars_meta.mat')

cars_train_annos = loadmat(devkit_path/'cars_train_annos.mat')

cars_test_annos = loadmat(devkit_path/'cars_test_annos.mat')
# Loading labels

labels = [c for c in cars_meta['class_names'][0]]

labels = pd.DataFrame(labels, columns=['labels'])

labels.head()
# Loading train data

frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']

df_train = pd.DataFrame(frame, columns=columns)

df_train['class'] = df_train['class']-1 # Python indexing starts on zero.

df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path

df_train['is_test'] = 0

df_train.head()
# Merging train data and labels

df_train = df_train.merge(labels, left_on='class', right_index=True)

df_train = df_train.sort_index()

df_train.head()
# Loading test data

frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]

columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']

df_test = pd.DataFrame(frame, columns=columns)

df_test['fname'] = [test_path/f for f in df_test['fname']] #  Appending Path

df_test['is_test'] = 1

df_test.head()
# Combine the train and test set

frames = [df_train, df_test]

df_labels = pd.concat(frames)

df_labels.reset_index(inplace=True, drop=True)

df_labels = df_labels[['fname', 'bbox_x1', 'bbox_y1','bbox_x2','bbox_y2','class', 'labels','is_test']]



# Add missing class name

df_labels.loc[df_labels['labels'].isnull(), 'labels'] = 'smart fortwo Convertible 2012'



# Adjust the test file names

#df_labels[df_labels["is_test"] == "1"] = 'test_' + df_labels['fname']

#df_labels['fname'][df_labels["is_test"] == "1"] = 'test_' + df_labels['fname']

#df_labels['fname'].loc[df_labels[df_labels["is_test"]]] = str('test_') + str(df_labels['fname'])

df_labels['fname'].loc[df_labels['is_test']==1] = str('test_') + str(df_labels['fname'])



# Add the cropped file names

df_labels['fname_cropped'] = df_labels['fname'].copy()

df_labels['fname_cropped'].loc[df_labels['is_test']==0] = str('cropped_') + str(labels_df['fname'])



# df_labels.to_csv('labels_with_annos.csv')

df_labels.head()
df_labels['bbox_h'] = (df_labels['bbox_y2'] - df_labels['bbox_y1']) + 1

df_labels['bbox_w'] = (df_labels['bbox_x2'] - df_labels['bbox_x1']) + 1

df_labels.head()
bb_hw = df_labels[['bbox_x1', 'bbox_y1','bbox_h', 'bbox_w']].values

hw_bb = df_labels[['bbox_h', 'bbox_w', 'bbox_x1', 'bbox_y1']].values

bb_hw
# Cars Distribution

freq_labels = df_train.groupby('labels').count()[['class']]

freq_labels = freq_labels.rename(columns={'class': 'count'})

freq_labels = freq_labels.sort_values(by='count', ascending=False)

freq_labels.head()



freq_labels.head(50).plot.bar(figsize=(15,10))

plt.xticks(rotation=90);

plt.xlabel("Cars");

plt.ylabel("Count");
df_train['labels'].unique()
df_train['class'].unique()
df_labels['labels'].nunique(), labels_df['class'].nunique()
def show_img(im, figsize=None, ax=None):

    if not ax: fig,ax = plt.subplots(figsize=figsize)

    ax.imshow(im)

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    return ax



# White text on black outline

def draw_outline(o, lw):

    o.set_path_effects([patheffects.Stroke(

        linewidth=lw, foreground='black'), patheffects.Normal()])

    

def draw_rect(ax, b):

#     patch = ax.add_patch(patches.Rectangle(xy, w,h, fill=False, edgecolor='white', lw=2))

    patch = ax.add_patch(patches.Rectangle(b[:2], b[3], b[2], fill=False, edgecolor='white', lw=2))

    draw_outline(patch, 4)

    

def draw_text(ax, xy, txt, sz=14):

    text = ax.text(*xy, txt,

        verticalalignment='top', color='white', fontsize=sz, weight='bold')

    draw_outline(text, 1)
def compare_top_losses(k, interp, labels_df, num_imgs):

    tl_val,tl_idx = interp.top_losses(k)

    classes = interp.data.classes

    probs = interp.probs

    columns = 2

    rows = 2

    

    topl_idx = 0   

    for i,idx in enumerate(tl_idx):

        fig=plt.figure(figsize=(10, 8))

        columns = 2

        rows = 1

        

        # Actual Image

        act_im, cl = interp.data.valid_ds[idx]

        cl = int(cl)        

        act_cl = classes[cl]

        act_fn = labels_df.loc[labels_df['class_name'] == act_cl]['filename'].values[0]

        

        # Predicted Image

        pred_cl = interp.pred_class[idx]

        pred_cl = classes[pred_cl]

        pred_fn = labels_df.loc[labels_df['class_name'] == pred_cl]['filename'].values[0]

        

        print(f'PREDICTION:{pred_cl}, ACTUAL:{act_cl}')

        print(f'Loss: {tl_val[i]:.2f}, Probability: {probs[i][cl]:.4f}')

              

        # Add image to the left column

        img_path = 'train/' + pred_fn

        im = plt.imread(path+img_path)

        fig.add_subplot(rows, columns, 1)

        plt.imshow(im)

        

        # Add image to the right column, need to change the tensor shape (permute) for matplotlib

        perm = act_im.data.permute(1,2,0)

        fig.add_subplot(rows, columns, 2)

        plt.imshow(perm)



        plt.show()
def compare_most_confused(most_confused, labels_df, num_imgs, rank):

    c1 = most_confused[:][rank][0]

    c2 = most_confused[:][rank][1]

    n_confused = most_confused[:3][0][1]

    print(most_confused[:][rank])

      

    # set the list of 

    f_1 = labels_df.loc[labels_df['class_name'] == c1]['filename'].values

    f_2 = labels_df.loc[labels_df['class_name'] == c2]['filename'].values



    fig=plt.figure(figsize=(10, 8))

    columns = 2

    rows = num_imgs

    for i in range(1, columns*rows +1, 2):

        # Add image to the left column

        img_path = 'train/' + f_1[i]

        im = plt.imread(path+img_path)

        fig.add_subplot(rows, columns, i)

        plt.imshow(im)

        

        # Add image to the right column

        img_path = 'train/' + f_2[i]

        im = plt.imread(path+img_path)

        fig.add_subplot(rows, columns, i+1)

        plt.imshow(im)



    plt.show()
help(get_transforms)
SZ = 224

SEED = 42

LABEL = 'labels'



car_tfms = get_transforms()



df_trn_labels = df_labels.loc[df_labels['is_test']==0, ['fname', 'labels', 'class']].copy()



src = (ImageList.from_df(df_trn_labels, train_path, cols='fname')

                .random_split_by_pct(valid_pct=0.2, seed=SEED)

                .label_from_df(cols=LABEL))



data = (src.transform(car_tfms, size=SZ)

            .databunch()

            .normalize(imagenet_stats))
arch = models.resnet152
car_tfms = get_transforms()



trn_df = train_df.loc[train_df, ['fname', 'class', 'labels']].copy()

trn_df.head()



"""

src = (ImageItemList.from_df(trn_df, train_path, cols='fname')

                    .random_split_by_pct(valid_pct=0.2, seed=42)

                    .label_from_df(cols='class_name'))



data = (src.transform(car_tfms, size=224)

            .databunch()

            .normalize(imagenet_stats))



data.batch_size = 32

data.batch_size

"""