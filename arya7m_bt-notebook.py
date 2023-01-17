from fastai import *

from fastai.vision import *

import os
%reload_ext autoreload 

%autoreload 2

%matplotlib inline
bs = 64

np.random.seed(2)
data_path = "../input/btumour"

#os.listdir(data_path)

data = ImageDataBunch.from_folder(data_path, bs=bs//4, size=299, ds_tfms=get_transforms(),num_workers=0,test='test').normalize(imagenet_stats)
data.show_batch(rows=5,figsize=(8,8))


learn = create_cnn(data, models.resnet101, metrics=accuracy, path=".")
learn.fit_one_cycle(20)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
interp.plot_top_losses(9,figsize=(4,4))
interp.most_confused(min_val = 0)
learn.save("stage-1")
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(15,max_lr=slice(1e-5,1e-3))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8))
learn.save("p-1")