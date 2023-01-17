from fastai.vision import *

from fastai.metrics import error_rate,accuracy
import os

import numpy as np
path_data = os.path.basename('/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)')

np.random.seed(42)

size = 224

tfms = get_transforms(do_flip=True,flip_vert=True, max_zoom=1.05, max_warp=0.2)
data = ImageDataBunch.from_folder('/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/color',

                                  ds_tfms=tfms,

                                  test='/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/Test',

                                  valid_pct=0.2,

                                 size=size,

                                 bs =64 )
data

arch = models.resnet50
learn = cnn_learner(data, arch, metrics=[accuracy,error_rate],model_dir="/kaggle/working/")
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(5, slice(min_grad_lr))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(2, slice(min_grad_lr))


interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
interp.most_confused(min_val = 2)
learn.export('/kaggle/working/resnet50leaf98.pkl')