from fastai import *

from fastai.vision import *

from tqdm import tqdm_notebook as tqdm



import random

import numpy as np

import keras

from random import shuffle

from keras.utils import np_utils

from shutil import unpack_archive

import matplotlib.pyplot as plt

import tensorflow as tf



%reload_ext autoreload

%autoreload 2

%matplotlib inline





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
!pip install imutils
import os

path="/kaggle/input/indian-sign-language/ISL/Train/"

labels = os.listdir("/kaggle/input/indian-sign-language/ISL/Train/")

print("No. of labels: {}".format(len(labels)))

print("-----------------")

for label in labels:

    print("{}, {} files".format(label, len(os.listdir("/kaggle/input/indian-sign-language/ISL/Train/"+label))))
# import numpy as np

# import matplotlib.pyplot as plt

# from PIL import Image



# fig, ax = plt.subplots(nrows=2, ncols=10)

# fig.tight_layout()

# cnt = 0

# for row in ax:

#     for col in row:

#         image_name = np.random.choice(os.listdir(path + labels[cnt]))

#         im = Image.open(path+"{}/{}".format(labels[cnt],image_name))

#         col.imshow(im)

#         col.set_title(labels[cnt])

#         col.axis('off')

#         cnt += 1

# plt.show()
size = 224

bs = 64

path="/kaggle/input/indian-sign-language/ISL/Train/"

data = ImageDataBunch.from_folder(path, 

                                  ds_tfms=get_transforms(do_flip=True, flip_vert=False),

                                  valid_pct=0.2, 

                                  size=size, 

                                  num_workers=4,

                                  bs=bs)

data

data.normalize(imagenet_stats)
# import glob

# path="/kaggle/input/indian-sign-language/ISL/Train/"

# np.random.seed(42)

# data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,

#                                   ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

# data
data.show_batch(rows=20, figsize=(15,15))
from fastai.metrics import error_rate 



thresh=0.2

learn = cnn_learner(data, models.vgg16_bn , metrics=[accuracy, error_rate])
learn.model_dir="/kaggle/working/models"
learn.lr_find()

learn.recorder.plot()
#learn.fit_one_cycle(6, max_lr=slice(1e-05, 1e-04))

learn.fit_one_cycle(6,1e-2)

learn.save('Sign-detection-stage-2')
learn.validate()

learn.show_results(ds_type=DatasetType.Train, rows=3, figsize=(20,20))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,15))
from fastai.vision import *

from fastai.metrics import accuracy

learner = create_cnn(data, models.resnet18, metrics=[accuracy], callback_fns=ShowGraph)
learner.model_dir='/kaggle/working/'

learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(8, max_lr=slice(1e-4, 1e-3))
interpreter = ClassificationInterpretation.from_learner(learner)

interpreter.most_confused(min_val=2)
learner.save("stage-1")

#learner.load(‘stage-1’)

learner.unfreeze()

learner.fit_one_cycle(12, max_lr=slice(1e-6, 1e-7))
from fastai.widgets import *



ds, idxs = DatasetFormatter().from_toplosses(learner)

ImageCleaner(ds, idxs, "/kaggle/working/")
interpreter.plot_top_losses(9, figsize=(15,15))
from fastai.widgets import *

ds, idxs = DatasetFormatter().from_similars(learner, layer_ls=[0,7,1], pool=None)
# # ImageCleaner(ds, idxs, '/kaggle/working/')

# most_unsure = DatasetFormatter.from_most_unsure(learner)

# wgt = PredictionsCorrector(*most_unsure)

# wgt.show_corrections(ncols=6, figsize=(9, 7))
interpreter.plot_confusion_matrix()
# learner.load(‘stage-1’)
# from keras.preprocessing.image import ImageDataGenerator

# train_data_dir="/kaggle/input/indian-sign-language/ISL/Train/"

# img_height=img_width= 224

# batch_size =50



# train_datagen = ImageDataGenerator(rescale=1./255,

#     rotation_range=60,

#     shear_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     validation_split=0.2) # set validation split



# train_generator = train_datagen.flow_from_directory(

#     train_data_dir,

#     target_size=(img_height, img_height),

#     batch_size=batch_size,

#     class_mode='binary',

#     subset='training') # set as training data



# validation_generator = train_datagen.flow_from_directory(

#     train_data_dir, # same directory as training data

#     target_size=(img_height, img_width),

#     batch_size=batch_size,

#     class_mode='binary',

#     subset='validation') # set as validation data



# model.fit_generator(

#     train_generator,

#     steps_per_epoch = train_generator.samples // batch_size,

#     validation_data = validation_generator, 

#     validation_steps = validation_generator.samples // batch_size,

#     epochs = nb_epochs)