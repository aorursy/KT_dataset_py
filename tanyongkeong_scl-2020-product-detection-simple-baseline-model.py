import os
import re
import json
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import glob
import cv2
import time

from fastai import *
from fastai.vision import *
import PIL
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, classification_report
# from sklearn.utils import class_weight
from tqdm import tqdm

tqdm.pandas()

# Import all dataset for word embedding
train_folder_dir = '../input/shopee-code-league-2020-product-detection/resized/train/'
test_folder_dir = '../input/shopee-code-league-2020-product-detection/resized/test/'

df_train = pd.read_csv("../input/shopee-code-league-2020-product-detection/train.csv")
df_test = pd.read_csv("../input/shopee-code-league-2020-product-detection/test.csv")

df_train.head()

df_train['cat'] = df_train['category']
df_train['cat'] = df_train['cat'].astype({"cat": str})
df_train['cat'] = df_train['cat'].apply(lambda x: x.zfill(2))
df_train['image_path'] = df_train["cat"] + "/" +df_train['filename']

train_image_paths = df_train['image_path'].values
test_image_paths = df_test['filename'].values

test_image_paths[0]
df_train["image_path"] = df_train["image_path"].apply(lambda x: train_folder_dir + x)
df = df_train[['image_path', 'category']]
df.rename(columns={'image_path': 'names', 'category':'labels'}, inplace=True)
df.head()
def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)
tfms = get_transforms(do_flip=True)
data = ImageDataBunch.from_df('', df, folder='', test= test_folder_dir, ds_tfms=tfms)
data.normalize()
print(data.classes)
len(data.classes),data.c
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet101, metrics=accuracy)
learn.loss_func = LabelSmoothingCrossEntropy()

learn.model
# Initial train model 
learn.fit_one_cycle(4)
learn.save('stage-1-101')
# Get the learning rate for next stage training
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()

# Retrain model using smaller learning rate
# If train loss < val loss then it is overfitted
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('stage-2-101')
# Retrain model using smaller learning rate
# If train loss < val loss then it is overfitted
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('stage-3-101')
# learn.load('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))

interp.most_confused(min_val=2)
preds,y = learn.TTA()
acc = accuracy(preds, y)
print('The validation accuracy is {} %.'.format(acc * 100))
preds, y = learn.get_preds(ds_type = DatasetType.Test)

filename_list = list(df_test.filename)
preds1 = np.argmax(preds, 1)
pred_list = [data.classes[int(x)] for x in preds1]

pred_dict = dict((key, value) for (key, value) in zip(learn.data.test_ds.items,pred_list))
pred_ordered = [pred_dict[Path(test_folder_dir + f)] for f in filename_list]
submissions = pd.DataFrame({'filename':filename_list,'category':pred_ordered})

submissions['category'] = submissions['category'].apply(lambda x: str(x).zfill(2))

submissions.to_csv("submission_transferLearning_{}.csv".format('noisyData'),index = False)
submissions.head()

