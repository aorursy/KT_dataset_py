import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')

import random

import os

import glob

import cv2 

from fastai.vision import *

from fastai import *

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import torch

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import plot_confusion_matrix





device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')

# Set seed fol all

def seed_everything(seed=1358):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
PATH = Path('../input/planets-dataset/planet/planet/')

train_img = PATH/'train-jpg'

train_folder = 'train-jpg'

test_img = PATH/'test-jpg'

model_dir = Path('/kaggle/working/')

bs = 64
PATH.ls()
train_df = pd.read_csv(os.path.join(PATH, 'train_classes.csv'))

# adding path to the image in our dataframe. 

train_df['image_name'] = train_df['image_name'].apply(lambda x: f'{train_folder}/{x}.jpg')

train_df.head()
# Since this is a multi lable task and the labels are given as tags in a single dataframe series

biner = MultiLabelBinarizer()

tags = train_df['tags'].str.split()

y = biner.fit_transform(tags)



labels = biner.classes_

print('Number of labels: ', len(labels))

print(labels)
# Getting the labels into one hot encoded form for EDA ease. 

for label in labels:

    train_df[label] = train_df['tags'].apply(lambda x: 1 if label in x.split()  else 0)

    

train_df.head()
train_df[labels].sum().sort_values(ascending=False).plot(kind='barh', figsize=(8,8))
df_asint = train_df.drop(train_df.columns[[0,1]], axis=1).astype(int)

coocc_df = df_asint.T.dot(df_asint)



coocc_df
# Confusion matrix. 
#reading images



random_imgs = train_df.ix[random.sample(list(train_df.index), 10)][['image_name', 'tags']]



to_read = random_imgs.loc[:, 'image_name'].values

tags = random_imgs.loc[:, 'tags'].values



images = [cv2.imread(os.path.join(PATH/file)) for file in to_read]

print("Number of images: ", len(images))

print("Size of an image: ", images[0].shape)
plt.figure(figsize=(25,15))

columns = 5

for i, image in enumerate(images):

    plt.subplot(len(images) / columns + 1, columns, i + 1)

    plt.imshow(image)

    plt.grid(False)

    plt.title(tags[i])
print(f"Size of Training set images: {len(list(train_img.glob('*.jpg')))}")

print(f"Size of Test set images: {len(list(test_img.glob('*.jpg')))}")

img_size = 128



tfms = get_transforms(do_flip=True,flip_vert=True,p_lighting=0.4,

                      max_lighting=0.3, max_zoom=1.05, max_rotate=360, xtra_tfms=[flip_lr()])





# The datablock API makes things very easy. 

# Im using 1% of the training data to validate the models. 



src = (ImageList.from_df(train_df, PATH, cols='image_name')

        .split_by_rand_pct(valid_pct=0.1)

        .label_from_df(label_delim=' '))





data = (src.transform(tfms,size=img_size,resize_method=ResizeMethod.CROP)

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )
data
data.train_ds
data.valid_ds
data.train_ds[0]
print(data.train_ds.y[200])

data.train_ds.x[200]
data.show_batch(rows=2)
model_1 = models.resnet50

acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)



learn = create_cnn(data, model_1, metrics=[acc_02, f_score], model_dir='/kaggle/working')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 0.01)
learn.model_dir = '/kaggle/working'

learn.save('resnet50-stage1')
model_2 = models.densenet121

learn_dense = create_cnn(data, model_2, metrics=[acc_02, f_score], model_dir='/kaggle/working')
learn_dense.fit_one_cycle(5, 0.01)
learn_dense.save('DenseNet121-stage1')
learn_dense.unfreeze()

learn_dense.lr_find()

learn_dense.recorder.plot()
learn_dense.fit_one_cycle(5, slice(1e-5, 1e-4))
learn_dense.save('DenseNet121-stage2')
data_2 = (src.transform(tfms,size=256,resize_method=ResizeMethod.CROP)

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )



data_2
model_2 = models.densenet121

learn_dense_2 = create_cnn(data_2, model_2, metrics=[acc_02, f_score], model_dir='/kaggle/working')
learn_dense_2.load('DenseNet121-stage2')
learn_dense_2.lr_find()

learn_dense_2.recorder.plot()
learn_dense_2.fit_one_cycle(10, 0.01)
learn_dense_2.unfreeze()

learn_dense_2.lr_find()

learn_dense_2.recorder.plot()
learn_dense_2.fit_one_cycle(10, slice(1e-5,1e-4))