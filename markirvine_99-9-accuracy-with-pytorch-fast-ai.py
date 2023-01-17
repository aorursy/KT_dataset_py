# Imports 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *

from fastai.metrics import error_rate



import os
# Define dataset path

data_path = Path("/kaggle/input/gtsrb-german-traffic-sign/")



# Put data into databunch using the folder names to assign the labels, 

# standard fast.ai image transforms not flipping the images

# rescale images to 244x244

# batch size of 64

# validation set containing 20% of the images

# normalization



data = ImageDataBunch.from_folder(data_path/"train",

                                  ds_tfms=get_transforms(do_flip=False),

                                  size=244,

                                  bs=64,

                                  valid_pct=0.2,

                                  seed=50).normalize(imagenet_stats)
#Define cnn based on resnet34 architecture and using an error rate metric to guage success.



learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir=Path("/kaggle/working/model")).to_fp16()
# Find best learning rates to train model on, should roughly use the region of steepest negitive gradient 

learn.lr_find()

learn.recorder.plot()
#Train the network with the given maximum learning rates

learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-03, 1e-01))