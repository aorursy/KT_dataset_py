%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
PATH = '../input/aptos2019-blindness-detection/'
train_df = pd.read_csv(PATH+'train.csv')

test_df = pd.read_csv(PATH+'test.csv')
train_df['id_code'] = train_df['id_code'].apply(lambda x : x+'.png')

test_df['id_code'] = test_df['id_code'].apply(lambda x : x+'.png')
train_df.head()
bs = 32

SIZE = 224



tfms = get_transforms(do_flip=True, flip_vert=True, max_warp=0., max_rotate=360.0)
data = (ImageList.from_df(df=train_df,folder='train_images',path=PATH)

       .split_by_rand_pct(0.2)

       .label_from_df(cols='diagnosis', label_cls=FloatList)

       .transform(tfms,size=SIZE)

       .databunch(bs=bs)

       .normalize(imagenet_stats))
data
data.show_batch(rows=3, fig_size=(5,5))
!mkdir -p '/tmp/.cache/torch/checkpoints/'

!cp ../input/resnet101/resnet101.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth
arch = models.resnet101
def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat),y,weights="quadratic"),device="cuda:0")
learn = cnn_learner(data, arch, metrics=[quadratic_kappa], pretrained=True, path='/models/')
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr

lr
from fastai.callbacks import SaveModelCallback

learn.fit_one_cycle(4, max_lr=slice(1e-3, 1e-1),callbacks=[SaveModelCallback(learn, every='epoch',  

                  monitor='quadratic_kappa', name='saved_net')])
learn.recorder.plot_losses()
learn.export('state1')

learn.save('state1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = 1e-6

learn.fit_one_cycle(4, max_lr=slice(lr,lr/10))
learn.save('state2')

learn.export('state2')
test = ImageList.from_df(test_df,folder='test_images',path=PATH)
SIZE = 256



data = (ImageList.from_df(df=train_df,folder='train_images',path=PATH)

       .split_by_rand_pct(0.2)

       .label_from_df(cols='diagnosis', label_cls=FloatList)

       .add_test(test)

       .transform(tfms,size=SIZE)

       .databunch(bs=bs)

       .normalize(imagenet_stats))
data
learn.data = data
learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.recorder.plot_lr()
lr = learn.recorder.min_grad_lr

lr
learn.fit_one_cycle(4, max_lr=lr)
learn.recorder.plot_losses()
learn.save('state3')

learn.export('state3')
valid_preds, valid_y = learn.TTA(ds_type=DatasetType.Valid)
test_preds, _ = learn.TTA(ds_type=DatasetType.Test)
# Thanks to Abhishek Thakur for this :)



import scipy as sp



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
print("coefficients:", coefficients)

print("validation score:", valid_score)
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

sample.diagnosis = test_predictions

sample.head()
sample.to_csv("submission.csv", index=None)