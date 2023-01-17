from fastai import *

from fastai.vision import *



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve



import os

print(os.listdir("../input"))
path = Path('../input/IDC_regular_ps50_idx5/')
fnames=get_files(path, recurse=True)
pattern= r'([^/_]+).png$'
tfms=get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)
data = ImageDataBunch.from_name_re(path, fnames, pattern, ds_tfms=tfms, size=50, bs=64,num_workers=2

                                  ).normalize()
data.show_batch(rows=3, figsize=(8,8))
learner= create_cnn(data, models.resnet18, metrics=[accuracy], model_dir='/tmp/models/')
learner.lr_find()

learner.recorder.plot()
lr=1e-2

learner.fit_one_cycle(6, lr)
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(10,slice(1e-5,1e-4))
learner.recorder.plot_losses()
learner.save('stage-1', return_path=True) 
learner.freeze_to(-3)
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(2,slice(1e-5,1e-4))
learner.freeze_to(-2)

learner.fit_one_cycle(2,slice(1e-5,1e-4))
conf= ClassificationInterpretation.from_learner(learner)

conf.plot_confusion_matrix(figsize=(10,8))
# Predictions of the validation data

preds_val, y_val=learner.get_preds()
#  ROC curve

fpr, tpr, thresholds = roc_curve(y_val.numpy(), preds_val.numpy()[:,1], pos_label=1)



#  ROC area

pred_score = auc(fpr, tpr)

print(f'ROC area is {pred_score}')
plt.figure()

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")