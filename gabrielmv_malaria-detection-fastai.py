import numpy as np 

import pandas as pd 

from pathlib import Path



from fastai import *

from fastai.vision import *



import os
data_path = Path("../input/cell_images/cell_images/")
data_path
transforms = get_transforms(do_flip = True, 

                            flip_vert = True, 

                            max_rotate = 10.0, 

                            max_zoom = 1.1, 

                            max_lighting = 0.2, 

                            max_warp = 0.2, 

                            p_affine = 0.75, 

                            p_lighting = 0.75)
data = ImageDataBunch.from_folder(data_path,

                                  train = '.',

                                  valid_pct = 0.2,

                                  size = 224,

                                  bs = 16,

                                  ds_tfms = transforms

                                 ).normalize(imagenet_stats)
data.classes
data.show_batch(rows = 4, figsize = (7, 7))
learn = cnn_learner(data, models.densenet161 , metrics = [accuracy, error_rate], model_dir = '/tmp/model/')
learn.lr_find()



learn.recorder.plot(suggestion = True)
min_grad_lr = learn.recorder.min_grad_lr



learn.fit_one_cycle(20, min_grad_lr)
learn.save('first-phase')
learn.unfreeze()



learn.lr_find()



learn.recorder.plot(suggestion = True)
min_grad_lr = learn.recorder.min_grad_lr



learn.fit_one_cycle(30, min_grad_lr)
learn.save('second-phase')
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize = (15, 10))
interp.plot_confusion_matrix()