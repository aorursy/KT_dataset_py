# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!rm -rf input
!cp -r ../input .
from fastai import *

from fastai.vision import *

import matplotlib.pyplot as plt
path = Path('input/train/')
np.random.seed(42)

data = ImageDataBunch.from_folder(path,test='../test', ds_tfms=get_transforms(),valid_pct=0.25,size=299,bs=32,num_workers=0)

data.normalize(imagenet_stats)
print(data.classes)

len(data.classes),data.c
data.show_batch(rows=3,figsize=(7,6))
learn = create_cnn(data,models.resnet50,metrics=error_rate)
from torch import nn

import torch.nn.functional as F



class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduction = reduction



    def forward(self, inputs, targets):

        if self.logits:

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        else:

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduction is None:

            return F_loss

        else:

            return torch.mean(F_loss)
learn.loss_fn = FocalLoss()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(7,slice(1e-2))
learn.save('stg-1')
learn.lr_find()

learn.recorder.plot()
#learn.load('stg-1')
learn.unfreeze()

learn.fit_one_cycle(6,max_lr=slice(1e-4,1e-3))
data = ImageDataBunch.from_folder(path,test='../test', ds_tfms=get_transforms(),valid_pct=0.25,size=350,bs=32,num_workers=0)

data.normalize(imagenet_stats)
interp = ClassificationInterpretation.from_learner(learn)

from sklearn import metrics

print(metrics.classification_report(interp.y_true.numpy(), interp.pred_class.numpy(),target_names =data.classes))
learn.save('stg-2')
learn.data=data
learn.unfreeze()

learn.fit_one_cycle(3,max_lr=slice(1e-5,1e-4))
preds,y=learn.get_preds(ds_type=DatasetType.Test)
preds = np.argmax(preds, axis = 1)

preds_classes = [data.classes[i] for i in preds]
submission = pd.DataFrame({ 'file': os.listdir('input/test'), 'species': preds_classes })

submission.to_csv('test_classification_results.csv', index=False)
submission
!rm -rf input
#data = ImageDataBunch.from_folder(path,test='../test', ds_tfms=get_transforms(),valid_pct=0.25,size=299,bs=32,num_workers=0)

#data = ImageDataBunch.from_csv(path, val_idxs =[0], test_name='test')

#learn = ConvLearner.pretrained(model, data, precompute=True)
def change_fc_data(learn, train_index, val_index):

    tmpl = f'_{learn.models.name}_{learn.data_.sz}.bc'

    names = [os.path.join(learn.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]

    learn.get_activations()

    act, val_act, test_act = [bcolz.open(p) for p in names]

    data_x = np.vstack([val_act, act])

    data_y = np.array(list(learn.data.val_y) + list(learn.data.trn_y))

    train_x = data_x[train_index] 

    valid_x = data_x[val_index]

    train_y = data_y[train_index] 

    valid_y = data_y[val_index]

    learn.fc_data = ImageClassifierData.from_arrays(learn.data_.path,

                 (train_x, train_y), 

                 (valid_x, valid_y), learn.data_.bs, classes=learn.data_.classes,

                 test = test_act if learn.data_.test_dl else None, num_workers=8)

    return learn
#ind = pd.read_csv(f'{PATH}/labels.csv', index_col='id')

#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

#for train_index, val_index in skf.split(ind.index, ind['breed'])



#data = ImageClassifierData.from_csv(val_idxs =[0], test_name='test')

#learn = ConvLearner.pretrained(model, data, precompute=True)

#learn = change_fc_data(learn, train_index, val_index)


