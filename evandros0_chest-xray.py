import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt



from fastai.vision import *

from fastai.metrics import error_rate



import os
path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/')
!ls {path}
tfms = get_transforms(do_flip=False)
# batch size

bs = 64
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, valid='test', bs=bs, size=224).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,6))
data.path = pathlib.Path('.')
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(3)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
tn, fp, fn, tp = interp.confusion_matrix().ravel()
precision = tp/(tp+fp)

recall = tp/(tp+fn)

specificity = tn/(tn+fp)

accuracy = (tp + tn)/(tn + fp + fn + tp)



print('''

        Precision: {:.3f}

        Recall: {:.3f}

        Specificity: {:.3f}

        Accuracy: {:.3f}

      '''.format(precision, recall, specificity, accuracy))
learn.unfreeze()
learn.fit_one_cycle(2)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
tn, fp, fn, tp = interp.confusion_matrix().ravel()
precision = tp/(tp+fp)

recall = tp/(tp+fn)

specificity = tn/(tn+fp)

accuracy = (tp + tn)/(tn + fp + fn + tp)



print('''

        Precision: {:.3f}

        Recall: {:.3f}

        Specificity: {:.3f}

        Accuracy: {:.3f}

      '''.format(precision, recall, specificity, accuracy))