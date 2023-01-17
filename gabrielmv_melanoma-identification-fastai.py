import numpy as np 

import pandas as pd 

from pathlib import Path



from fastai import *

from fastai.vision import *

from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback



import os
os.listdir('../input/dermmel/DermMel/')
data_path = Path('../input/dermmel/DermMel/')
transforms = get_transforms(do_flip = True, 

                            flip_vert = True, 

                            max_rotate = 355.0, 

                            max_zoom = 1.5, 

                            max_lighting = 0.3, 

                            max_warp = 0.2, 

                            p_affine = 0.75, 

                            p_lighting = 0.75)
data = ImageDataBunch.from_folder(data_path,

                                  valid_pct = 0.15,

                                  size = 200,

                                  bs = 64,

                                  ds_tfms = transforms

                                 )



data.normalize(imagenet_stats)
data.classes
data.show_batch(rows = 5, figsize = (12, 12))
learn = cnn_learner(data, models.resnet152 , metrics = [accuracy], model_dir = '/tmp/model/')
reduce_lr_pateau = ReduceLROnPlateauCallback(learn, patience = 10, factor = 0.2, monitor = 'accuracy')



#early_stopping = EarlyStoppingCallback(learn, monitor = 'accuracy', patience = 6)



save_model = SaveModelCallback(learn, monitor = 'accuracy', every = 'improvement')



callbacks = [reduce_lr_pateau, save_model]
learn.unfreeze()



learn.lr_find()



learn.recorder.plot(suggestion = True)
min_grad_lr = learn.recorder.min_grad_lr



learn.fit_one_cycle(100, min_grad_lr, callbacks = callbacks, wd = 1e-3)
learn.save('model')
learn.load('model')
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize = (12, 12), heatmap = False)
interp.most_confused()
predictions, y, loss = learn.get_preds(with_loss = True)



acc = accuracy(predictions, y)
print('Accuracy: {0}'.format(acc))