%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import gc

from subprocess import check_output

from PIL import Image

from matplotlib import image

from matplotlib import pyplot

from fastai import *

from fastai.vision import *
#check file directory structure

print(check_output(["ls", "../input/chest_xray/chest_xray/train"]).decode("utf8"))
#return array of labels for a given set of images

def get_labels(label,size):

    return np.array([label] * size)



#creates and returns shuffled arrays 

def shuffle_dataset(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]
#specify paths to images

PATH        = '../input/chest_xray/chest_xray'



#training set images

TRAIN_NORM  = PATH + '/train/NORMAL/'

TRAIN_PNEU  = PATH + '/train/PNEUMONIA/'
files_normal  = get_image_files(TRAIN_NORM)

files_pneu    = get_image_files(TRAIN_PNEU)



#merge both classes into single list

data_set = files_normal + files_pneu



#create arrays of labels

data_labels = np.concatenate((get_labels('normal',len(files_normal)), get_labels('pneumonia', len(files_pneu))), axis=0)



#return shuffled datasets

data_shuffled, data_labels = shuffle_dataset(np.array(data_set), data_labels)
#partition data into training and validation sets

data = ImageDataBunch.from_lists(PATH+'/train/', data_shuffled, data_labels, ds_tfms=get_transforms(), size=264)



#normalize pixel values

data.normalize(imagenet_stats)



#view a small sample of the data

data.show_batch(rows=3, figsize=(12,10))
#initialize convolutional neural net

learner = cnn_learner(data, models.resnet34, metrics=error_rate)



#view the CNN model

learner.model
#fit the model

learner.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learner)



#show the images that were the furthest from correct

interp.plot_top_losses(9, figsize=(15,10))



# plot true/false positives/negatives

interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
# use for garbage collection, if needed

# del data, interp, learner

# gc.collect()