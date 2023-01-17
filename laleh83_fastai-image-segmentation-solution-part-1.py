%reload_ext autoreload

%autoreload 2

%matplotlib inline
# import required libraries

import numpy as np 

import pandas as pd 

from fastai import *

from fastai.vision import *

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing import image

from pathlib import Path

import os

import glob  # used for loading multiple files

# import PIL 

!mkdir -p /tmp/.cache/torch/checkpoints/

# !cp /input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
# root directory

path = Path("../input/severstal-steel-defect-detection");

path.ls()
path_train_img = path/'train_images';

path_test_img = path/'test_images';
# get a list of filenames in the train image directory

fnames = get_image_files(path_train_img)

fnames[:5]
# labels in the train.csv file  

train = pd.read_csv(path/'train.csv')

train.head()
# function to plot an image

def plot_img(ImageId):

    img_id = ImageId+'.jpg'

    img = open_image(str(path_train_img) + '/'+img_id)

    return img



# function to plot a mask of an image

def plot_mask(ImageId_ClassId):

    mask = open_mask_rle(train.loc[lambda df: df["ImageId_ClassId"] == ImageId_ClassId, "EncodedPixels"].values[0], shape=(256, 1600))

    mask = ImageSegment(mask.data.transpose(2, 1))

    return mask



def plot_img_mask(ImageId,ClassId):

    defect_img = plot_img(ImageId)

    defect_mask = plot_mask(ImageId+'.jpg_'+str(ClassId))

    defect_img.show(y=defect_mask, figsize=(20, 10), title = 'image & its masks')
plot_img('0002cc93b')
plot_mask('0002cc93b.jpg_1')
plot_img('fff02e9c5')
plot_mask('fff02e9c5.jpg_3')
plot_img('000f6bf48')
plot_mask('000f6bf48.jpg_4')
# Visualize mask and image in one plot

plot_img_mask('000f6bf48',4)
# https://www.kaggle.com/mayurkulkarni/fastai-simple-model-0-88-lb

def train_pivot(train_csv):

    df = pd.read_csv(train_csv)



    def group_func(df, i):

        reg = re.compile(r'(.+)_\d$')

        return reg.search(df['ImageId_ClassId'].loc[i]).group(1)



    group = df.groupby(lambda i: group_func(df, i))



    df = group.agg({'EncodedPixels': lambda x: list(x)})



    df['ImageId'] = df.index

    df = df.reset_index(drop=True)



    df[[f'EncodedPixels_{k}' for k in range(1, 5)]] = pd.DataFrame(df['EncodedPixels'].values.tolist())

    

    df = df.drop(columns='EncodedPixels')

    train_df = df.fillna(value=' ')

    return train_df
train_df = train_pivot(str(path)+'/train.csv')

train_df.head()
# adding a flag to determine whether or not an image has defects

train_df['has_defects'] = 0

train_df.loc[(train_df['EncodedPixels_1'] != ' ') |  (train_df['EncodedPixels_2'] != ' ') | (train_df['EncodedPixels_3'] != ' ')

            | (train_df['EncodedPixels_4'] != ' '), 'has_defects'] = 1 
train_df.head()
print('There are' ,train_df[train_df.has_defects == 1].shape[0] , 'images with defects and' , train_df[train_df.has_defects == 0].shape[0]

      , 'without defects in the training set') 
# using the original dataframe where there are 4 lines for each ImageId, I calculate the number of defects for each image



tmp = train.copy()

tmp['ImageId'] = tmp['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

tmp['ClassId'] = tmp['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

tmp['has_defects'] = tmp.EncodedPixels.apply(lambda x: 1 if not pd.isnull(x) else 0)

defects = pd.DataFrame(tmp.groupby(by="ImageId")['has_defects'].sum())

defects.reset_index(inplace=True)  # convert the image_id which is an index to a column so that the dataframes can be joined on that

defects.rename(columns={"has_defects": "no_of_defects"},inplace=True) # rename the aggregated column ready for the join 

defects.head()





# add the no_of_defects to the labels dataframe



train_df = train_df.merge(defects, left_on='ImageId', right_on='ImageId', how='left')

train_df.head(4)
sns.countplot(train_df.no_of_defects)
train_df.no_of_defects.value_counts().sort_values(ascending=False)
# example of steel images more than 1 defects

train_df[train_df.no_of_defects > 1]
# an example image with 2 defects: class 3 & class 4

plot_img('fd26ab9ad')
plot_img_mask('fd26ab9ad','3')

plot_img_mask('fd26ab9ad','4')
train_clf = train_df[['ImageId','no_of_defects']] 

train_clf.head()
# creating the specific data format called ImageDataBunch required by the fastai models. It bundles the actual training images

# in the image directory with the labels we loaded into a dataframe 

np.random.seed(42)

bs = 64

data = ImageDataBunch.from_df(path_train_img, train_clf, ds_tfms=get_transforms(), size=256, bs=bs, test=path_test_img

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/kaggle/working")
learn.model
learn.fit_one_cycle(1)
learn.save('DefectClass_stage-1')
# !mkdir exports
# learn.export()
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.recorder.plot()
# ingest more data into the model to improve error_rate

learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-04,1e-03))
learn.save('DefectClass_stage-2')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-05,1e-04))
learn.save('DefectClass_stage-3')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.most_confused(min_val=2)
learn.predict(is_test=True)
learn.show_results()
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn)