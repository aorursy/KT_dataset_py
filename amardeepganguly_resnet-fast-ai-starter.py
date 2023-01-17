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




import os

import cv2

import random

import numpy as np

import pandas as pd

import scipy as sp

import torch

from fastai.vision import *

import glob

print(os.listdir("../input/fastai-pretrained-models"))

PATH = "../input/aptos2019-blindness-detection"

!mkdir -p /tmp/.cache/torch/checkpoints/

!cp ../input/fastai-pretrained-models/resnet101-5d3b4d8f.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth
SIZE=224



train_df=pd.read_csv(PATH+'/train.csv')

test_df=pd.read_csv(PATH+'/sample_submission.csv')
train = ImageList.from_df(train_df, path=PATH, cols='id_code', folder="train_images", suffix='.png')

test = ImageList.from_df(test_df, path=PATH, cols='id_code', folder="test_images", suffix='.png')
def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
def crop_image(img,tol=7):        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def open_aptos2019_image(fn, convert_mode, after_open)->Image:

    image = cv2.imread(fn)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = crop_image(image)

    image = cv2.resize(image, (SIZE, SIZE))

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image , (0,0) , SIZE/10) ,-4 ,128)

    return Image(pil2tensor(image, np.float32).div_(255))



vision.data.open_image = open_aptos2019_image
from sklearn.metrics import cohen_kappa_score

data = (

    train.split_by_rand_pct(0.2)

    .label_from_df(cols='diagnosis', label_cls=FloatList)

    .add_test(test)

    .transform(get_transforms(), size=SIZE)

    .databunch(path=Path('.'), bs=32).normalize(imagenet_stats)

)

data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet101, metrics=[quadratic_kappa], pretrained=True)

learn.fit_one_cycle(6, slice(2.75e-03,2.75e-02))

learn.recorder.plot_losses()

learn.recorder.plot_metrics()



learn.save('stage1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 2.75e-03

learn.fit_one_cycle(4,slice(1.5e-06,lr/8),wd=0.05)

#learn.fit_one_cycle(10, slice(5e-06, lr/8))
valid_preds, valid_y = learn.TTA(ds_type=DatasetType.Valid)

test_preds, _ = learn.TTA(ds_type=DatasetType.Test)
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(valid_preds, valid_y)

coefficients = optR.coefficients()



valid_predictions = optR.predict(valid_preds, coefficients)[:,0].astype(int)

test_predictions = optR.predict(test_preds, coefficients)[:,0].astype(int)



valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")
valid_score = cohen_kappa_score(valid_y.numpy().astype(int), valid_predictions, weights="quadratic")
print("coefficients:", coefficients)

print("validation score:", valid_score)
test_df.diagnosis = test_predictions

test_df.to_csv("submission.csv", index=None)

test_df.head()