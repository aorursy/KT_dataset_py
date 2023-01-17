# Mount the google drive to google colab

from google.colab import drive

drive.mount('/content/gdrive',force_remount=True)

root_path = "gdrive/My Drive/colab_folder/"
# Upload the kaggle API json file

from google.colab import files

files.upload()
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# Search datasets by name

!kaggle datasets list -s boat
!kaggle datasets download -d clorichel/boat-types-recognition -p /content/gdrive/My\ Drive/colab_folder/datasets
!pwd
import os

os.chdir('gdrive/My Drive/colab_folder/datasets')
!ls
!mkdir boats
!unzip -q boat-types-recognition.zip #difference between q and qq?
!ls
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.metrics import accuracy
bs=8 #16 was too much for memory
np.random.seed(12)

data= ImageDataBunch.from_folder(Path('boats/'), train='.',valid_pct=0.20,size=299, bs=bs, ds_tfms=get_transforms()).normalize(imagenet_stats)
data
data.show_batch(rows=3,figsize=(10,10))
data.classes
data.c, len(data.train_ds), len(data.valid_ds) #what is .train_ds
learn=cnn_learner(data,models.resnet152,metrics=accuracy)
learn.model
learn.fit_one_cycle(4) 
learn.save('stage-1')
interp=ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(10,10))
interp.most_confused(min_val=2)
interp.plot_confusion_matrix()
learn.load('stage-1'); 
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5,max_lr=slice(1e-6,1e-4)) 
learn.save('stage-2')
learn.load('stage-2');
interp=ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.lr_find()

learn.recorder.plot()
#learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))
learn.save('stage-3')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(10,10))
interp.plot_confusion_matrix()