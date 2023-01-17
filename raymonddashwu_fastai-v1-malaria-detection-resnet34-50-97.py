# Dataset taken from https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.metrics import error_rate

import numpy as np
torch.cuda.set_device(0)
torch.cuda.is_available(), torch.backends.cudnn.enabled
import fastai.utils.collect_env; fastai.utils.collect_env.show_install(1)
path = Path('../input/cell_images/cell_images'); path
path.ls()
import os
np.random.seed(42)
path_image_parasitized = Path('../input/cell_images/cell_images/Parasitized/')

path_image_uninfected = Path('../input/cell_images/cell_images/Uninfected/')
parasitized_images = get_image_files(path_image_parasitized)[:10]

parasitized_images
path_para, dirs, files = next(os.walk(path_image_parasitized))

file_count = len(files)

print("Number of images for parasitized:", file_count)
images_para = open_image(parasitized_images[np.random.randint(0, 9)])

images_para.show(figsize=(5,5))

images_para.size
uninfected_images = get_image_files(path_image_uninfected)[:10]

uninfected_images
path_unin, dirs, files = next(os.walk(path_image_uninfected))

file_count = len(files)

print("Number of images for uninfected:", file_count)
images_unin = open_image(uninfected_images[np.random.randint(0, 9)])

images_unin.show(figsize=(5,5))

images_unin.size
path
data = ImageDataBunch.from_folder(path, train=".",

                                  valid_pct=0.2, # Splits the dataset into 80/20% training/validation

                                  ds_tfms=get_transforms(do_flip = True, flip_vert = True, max_warp=0), # AFAIK these types of images can be flipped any direction vertically, horizontally, 90 degrees in actual cell images

                                  size=224,bs=64 # Trying out a larger 256 size at first

                                 ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(15,14))
print(data.classes)

len(data.classes),data.c
# Trying out mixed precision learning

# https://docs.fast.ai/callbacks.fp16.html

# https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

# Update: Did not work. Error: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same



learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/kaggle/model")
learn.model
learn.fit_one_cycle(4)
learn.save('/kaggle/working/malaria_resnet34_initial_training')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# Note: these are images that are in the top_losses section. Model was not sure what these were 

# TODO: Ask someone with biology background to predict these

interp.plot_top_losses(9, figsize=(15,14),  heatmap = False)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.save('/kaggle/working/malaria_resnet34_unfrozen_and_tuned')
# learn.load('malaria_unfrozen_and_tuned')
data = ImageDataBunch.from_folder(path, train=".",

                                  valid_pct=0.2, # Splits the dataset into 80/20% training/validation

                                  ds_tfms=get_transforms(flip_vert = True), # AFAIK images can be flipped any direction vertically, horizontally, 90 degrees in actual cell images

                                  size=299,bs=32

                                 ).normalize(imagenet_stats)
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-3/2))
learn.save('/kaggle/working/malaria_initial_training_resnet50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4, (1e-3)/5))



# learn.fit_one_cycle(8, 3e-3)

# https://youtu.be/ccMHJeQU4Qw?t=3552 ¯\_(ツ)_/¯
learn.recorder.plot_losses()
learn.save('/kaggle/working/malaria_resnet50_unfrozen_and_tuned')
# Wrongly implemented transfer learning
#learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/kaggle/model")
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(8, 3e-3)

# https://youtu.be/ccMHJeQU4Qw?t=3552 ¯\_(ツ)_/¯
# learn.save('/kaggle/working/malaria_initial_training_resnet50')
# learn.lr_find()

# learn.recorder.plot()
#TODO: Figure out how many epochs before this starts doing badly. Just starting to get good! Could push a couple more.



# learn.unfreeze()

# learn.fit_one_cycle(15, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,14), heatmap = False)



# Observation - some of these uninfected have the blob in the middle that make it look like an infected blood cell

# and vice versa. Some are clean that are labeled as infected.

# I wonder if these are mislabeled.
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)