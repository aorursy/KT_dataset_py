# Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#FastAI library

from fastai.vision import *

from fastai.metrics import error_rate, accuracy



import gc
# Load Data

path = '../input/image-dataset'    #path for image dataset



np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#View pre-defined classes in the dataset

data.classes
#Dataset batch stats

data.batch_stats
#View (random) data in the dataset

data.show_batch(rows=2, figsize=(8,8))
#Build Model

learner = create_cnn(data, models.resnet34, metrics = accuracy)
#Train Model

defaults.device = torch.device('cuda')  #checks if the gpu is used

learner.fit_one_cycle(5, max_lr=slice(3e-5,3e-4))
gc.collect()
interpret = ClassificationInterpretation.from_learner(learner)
interpret.confusion_matrix()
#Plot Top Losses

interpret.plot_top_losses(9, figsize=(15,15))