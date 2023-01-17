%reload_ext autoreload

%autoreload 2

%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



from sklearn.model_selection import train_test_split

from fastai.vision import *

from fastai.metrics import error_rate
SEED = 42

VALIDATION_PCT = 0.1

IMAGE_SIZE_64 = 64

IMAGE_SIZE_224 = 224

BATCH_SIZE_64 = 64

BATCH_SIZE_224 = 224

PATH = Path('../input')

TRAIN_PATH = PATH/'train'

TEST_FOLDER_PATH = "test/test"

SAMPLE_SUBMISSION_PATH = PATH/"sample_submission.csv"
def load_data(path, image_size, batch_size, validation_pct = VALIDATION_PCT):

    data = (ImageList.from_folder(path)

                .split_by_rand_pct(validation_pct, seed = SEED) # Taking 10% of data for validation set

                .label_from_folder() # Label the images according to the folder they are present in

                .transform(get_transforms(), size = image_size) # Default transformations with the given image size 

                .databunch(bs = batch_size) # Using the given batch size

                .normalize(imagenet_stats)) # Normalizing the images to improve data integrity

    return data
data_64 = load_data(TRAIN_PATH, IMAGE_SIZE_64, BATCH_SIZE_224, VALIDATION_PCT)

data_64
learn = cnn_learner(data_64, # training data with low resolution

                    models.resnet152, # Model which is pretrained on the ImageNet dataset 

                    metrics = [error_rate, accuracy], # Validation metrics

                    model_dir = '/tmp/model/') # Specifying a write location on the machine where the lr_find() can write 
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2)
test_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

test_df.head()

test_images = ImageList.from_df(test_df, PATH/"", folder = TEST_FOLDER_PATH)
test_images[1]

test_images[192]
data_256 = load_data(TRAIN_PATH, IMAGE_SIZE_224, BATCH_SIZE_64, VALIDATION_PCT)

data_256.add_test(test_images)
learn.data = data_256

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(30, max_lr = slice(1.0e-4, 1.0e-3))
test_probabalities, _ = learn.get_preds(ds_type=DatasetType.Test)

test_predictions = [data_256.classes[pred] for pred in np.argmax(test_probabalities.numpy(), axis=-1)]
test_df.predicted_class = test_predictions

test_df.to_csv("submission.csv", index=False)

test_df.head()