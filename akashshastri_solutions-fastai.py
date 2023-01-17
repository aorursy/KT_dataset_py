# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("/kaggle/input/export-fork1of4"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
from fastai.callbacks import *
base_image_dir = os.path.join('../', 'input/aptos2019-blindness-detection')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head()
tfms = get_transforms(do_flip=True,

                      flip_vert=True,

                      max_rotate=360,

                      max_warp=0.,

                      max_zoom=1.05,

                      max_lighting=0.1,

                      p_lighting=0.5

                     )
bs = 32 

sz=320

tfms = get_transforms()

src = (ImageList.from_df(df=df

                         ,path=''

                         ,cols='path'

                        ) 

        .split_by_rand_pct(0.20) 

        .label_from_df(cols='diagnosis',label_cls=FloatList) 

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') 

        .databunch(bs=bs,num_workers=4) 

        .normalize(imagenet_stats)      

       )
learn1 = load_learner('../input/pretrainblindness2/','final (11).pkl')
learn1.data = data
learn2 = load_learner('../input/pretrainblindness1/','final (10).pkl')
learn2.data = data
learn3 = load_learner('../input/exportdense/','final (8).pkl')
learn3.data = data
learn4 = load_learner('../input/exportmodel/','final.pkl')
learn4.data = data
interp1 = ClassificationInterpretation.from_learner(learn1)

losses1,idxs1 = interp.top_losses()
interp2 = ClassificationInterpretation.from_learner(learn2)

losses2,idxs2 = interp.top_losses()

interp3 = ClassificationInterpretation.from_learner(learn3)

losses3,idxs3 = interp.top_losses()

interp4 = ClassificationInterpretation.from_learner(learn4)

losses4,idxs4 = interp.top_losses()
valid_preds1 = learn1.get_preds(ds_type=DatasetType.Valid)

valid_preds2 = learn2.get_preds(ds_type=DatasetType.Valid)

valid_preds3 = learn3.get_preds(ds_type=DatasetType.Valid)

valid_preds4 = learn4.get_preds(ds_type=DatasetType.Valid)
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

from fastai.callbacks import *
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



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



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
optR1 = OptimizedRounder()

optR1.fit(valid_preds1[0],valid_preds1[1])

optR2 = OptimizedRounder()

optR2.fit(valid_preds2[0],valid_preds2[1])

optR3 = OptimizedRounder()

optR3.fit(valid_preds3[0],valid_preds3[1])

optR4 = OptimizedRounder()

optR4.fit(valid_preds4[0],valid_preds4[1])
coefficients1 = optR1.coefficients()

coefficients2 = optR2.coefficients()

coefficients3 = optR3.coefficients()

coefficients4 = optR4.coefficients()
print(coefficients1)

print(coefficients2)

print(coefficients3)

print(coefficients4)
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn1.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))

learn2.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))

learn3.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))

learn4.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))
preds1,_ = learn1.TTA(ds_type=DatasetType.Test)

preds2,_ = learn2.TTA(ds_type=DatasetType.Test)

preds3,_ = learn3.TTA(ds_type=DatasetType.Test)

preds4,_ = learn4.TTA(ds_type=DatasetType.Test)



labelled_preds = []

pred11 = preds4 + preds1 + preds2 + preds3

for pred in pred11:

    labelled_preds.append(int(np.argmax(pred))+1)



test_predictions = optR1.predict(labelled_preds, coefficients1)
sample_df.diagnosis = test_predictions.astype(int)

sample_df.groupby('diagnosis').count()
sample_df.to_csv('submission.csv',index=False)