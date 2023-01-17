# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# load libraries

from fastai import *

from fastai.vision import *

import pandas as pd
size = 96 # ssize of input images

bs = 32 # batch size

tfms = get_transforms(do_flip=False,flip_vert=True)
path = Path('../input/gender/gender')
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='Train',test='Testing',valid='Validation',

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
data.show_batch(rows=3)
path.ls()
model = models.resnet18
data.path = '/tmp/.torch/models'
learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.summary()
learn.lr_find()

learn.recorder.plot()
learn.save("stage-1")
lr = 2e-2
learn.fit_one_cycle(4,slice(lr))
learn.fit_one_cycle(4,slice(lr))
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(4,slice(lr))
accuracy(*learn.TTA())
learn.save("stage-2")
size = 128
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='Train',test='Testing',valid='Validation',

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
learn.data = data
learn.freeze()

learn.lr_find()

learn.recorder.plot()
lr = 1e-3
learn.fit_one_cycle(5,slice(lr))
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(5,slice(lr))
accuracy(*learn.TTA())
learn.save('stage-3')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
from fastai.vision import Image,pil2tensor

from PIL import Image

import cv2



def array2tensor(x):

    """ Return an tensor image from cv2 array """

    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

    return Image(pil2tensor(x,np.float32).div_(255))
! wget http://66.media.tumblr.com/f740c7cdd5f87b93005343d42bc11e4c/tumblr_nq5bz8wSdH1uyaaeio2_1280.jpg
! ls
img = cv2.imread('tumblr_nq5bz8wSdH1uyaaeio2_1280.jpg')

img = array2tensor(img)



learn.predict(img)