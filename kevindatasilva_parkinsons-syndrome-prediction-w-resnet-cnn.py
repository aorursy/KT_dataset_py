import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import * # import the FastAI v3 lib which includes pytorch

from fastai.vision import  * # import all of the computer vision related libs from vision 



# lets import our necessary magic libs

%reload_ext autoreload

%autoreload 2

%matplotlib inline
BATCH_SIZE = 64 

IMG_SIZE = 224

WORKERS = 0 

DATA_PATH_STR = '../input/parkinsons-drawings/'

DATA_PATH_OBJ = Path(DATA_PATH_STR)
tfms = get_transforms() # standard data augmentation ()



data = (ImageList.from_folder(DATA_PATH_OBJ)        # get data from path object

        .split_by_rand_pct()                        # separate 20% of data for validation set

        .label_from_folder()                          # label based on directory

        .transform(tfms, size=IMG_SIZE)                   # added image data augmentation

        .databunch(bs=BATCH_SIZE, num_workers=WORKERS)    # create ImageDataBunch

        .normalize(imagenet_stats))                   # normalize RGB vals using imagenet stats
# lets check to see if the seperations were done correctls

('training DS size:', len(data.train_ds), 'validation DS size:' ,len(data.valid_ds))
# lets check our labels/classes

data.classes
data.show_batch(rows=4, figsize=(10,8))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp/models')
# lets start training via one cycle 

learn.fit_one_cycle(1)

# should happen quickly since the dataset is relatively small
interp = ClassificationInterpretation.from_learner(learn)
# show me what the model was most confident in yet, was incorrect.

losses,idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
# lets unfreeze the remainder of the model to see if our model can do better 

learn.unfreeze()
learn.fit_one_cycle(3)
learn.save('stage-1-74')
# lets try to find the learning rate to improve the model accuracy

learn.lr_find()

learn.recorder.plot(suggestion=True)
# instead of the default max_lr

# lets pass our cycle the lowest learning rates suggested

learn.fit_one_cycle(10, max_lr=slice(1e-4,1e-6))
learn.save('stage-2-86')
interp = ClassificationInterpretation.from_learner(learn)

interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused()
learn5 = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/tmp/models')
# lets fit only our top layers to see how well she does 

learn5.fit_one_cycle(1)
# Lets optimize

learn5.lr_find()

learn5.recorder.plot(suggestion=True)
learn5.fit_one_cycle(2, max_lr=slice(1e-2,1e-4))
# lets now unfreeze the rest to see if we can improve

learn5.unfreeze()
# Lets optimize

learn5.lr_find()

learn5.recorder.plot(suggestion=True)
# now lets fitting with a suggested lr instead of default

learn5.fit_one_cycle(20, max_lr=slice(1e-3,1e-6))
interp50 = ClassificationInterpretation.from_learner(learn5)
interp50.most_confused()
learn5.save('stage-1-93')