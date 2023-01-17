import numpy as np

import pandas as pd

import os

from fastai import *

from fastai.vision import *
# Handy little function that comes stock on all Kaggle kernals

# I tend to revisit and tweek this many times to make sure I'm getting all the apropriate file paths right



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pix=250

#number of photos to download per class
classes = ['Bagels', 'Cupcakes','Doughnuts', 'Muffins']
folder = 'Bagels'

file = 'Bagels.txt'
path = Path('/kaggle/working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/image-urls-for-lesson-2/* {path}/
download_images(path/file, dest, max_pics=pix)

#The invalid URL warning this produces is for the excess blank newlines that I probably could have cleaned out of the URL list documents.
folder = 'Cupcakes'

file = 'Cupcake.txt'

path = Path('/kaggle/working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=pix)
folder = 'Doughnuts'

file = 'Doughnuts.txt'

path = Path('/kaggle/working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=pix)
folder = 'Muffins'

file = 'Muffin.txt'

path = Path('/kaggle/working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=pix)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path,

                                 train='.',

                                 valid_pct=0.3,

                                 ds_tfms=get_transforms(),

                                 size=224,

                                 bs=64,

                                 num_workers=0

                                 ).normalize(imagenet_stats)
data.classes
data.show_batch(row=3, figsize=(12,12))
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=error_rate)

Model_Path = Path('/kaggle/input/bcdmmodel/')

learn.model_dir = Model_Path

learn.load('stage-1')

print('Model Load Complete')
learn.unfreeze()
learn.fit_one_cycle(1)
Model_Path = Path('/kaggle/working')

learn.model_dir = Model_Path

learn.save('stage-2')
learn.lr_find()
learn.recorder.plot()
learn.save('stage-3')
learn.load('stage-3')

print('\n')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-3))#(2, max_lr=slice(1e-5,2e-2))
learn.save('stage-4')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(14,14))
from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ImageCleaner(ds, idxs, path)
#ds, idxs = DatasetFormatter().from_similars(learn)

#ImageCleaner(ds, idxs, path, duplicates=True)
defaults.device = torch.device('cpu') # For inferance a cpu is sufficient for most use cases

                                      # A GPU would be overkill unless this was scaled up to alot of simultanious images
img = open_image(Path('/kaggle/working/Cupcakes/00000008.jpg'))

img
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
# To export the model in a .pkl format, which would be useful if we wanted to host this model on another platform.

# webapp hosting isn't something thats optimal to show off though Kaggle so this is where I'll wrap up this notebook.



learn.export()