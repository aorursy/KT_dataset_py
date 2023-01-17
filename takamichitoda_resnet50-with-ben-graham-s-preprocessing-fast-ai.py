%reload_ext autoreload

%autoreload 2



import os

import cv2

import random

import numpy as np

import pandas as pd

import scipy as sp

from sklearn.metrics import cohen_kappa_score



import torch

from fastai.vision import *
SEED = 1234

SIZE = 224



PATH = "../input/aptos2019-blindness-detection"
def seed_everything(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything(SEED)
!mkdir -p /tmp/.cache/torch/checkpoints/

!cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth
train_df = pd.read_csv(PATH+"/train.csv")

sub = pd.read_csv(PATH+"/sample_submission.csv")
train = ImageList.from_df(train_df, path=PATH, cols='id_code', folder="train_images", suffix='.png')

test = ImageList.from_df(sub, path=PATH, cols='id_code', folder="test_images", suffix='.png')
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
data = (

    train.split_by_rand_pct(0.2, seed=SEED)

    .label_from_df(cols='diagnosis', label_cls=FloatList)

    .add_test(test)

    .transform(get_transforms(), size=SIZE)

    .databunch(path=Path('.'), bs=64).normalize(imagenet_stats)

)

data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet50, metrics=[quadratic_kappa], pretrained=True)

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, 1e-2)

learn.recorder.plot_losses()

learn.recorder.plot_metrics()
valid_preds, valid_y = learn.get_preds(ds_type=DatasetType.Valid)

test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# ref: https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa

# thank you Abhishek Thakur!!

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
sub.diagnosis = test_predictions

sub.to_csv("submission.csv", index=None)

sub.head()
sub.diagnosis.hist();