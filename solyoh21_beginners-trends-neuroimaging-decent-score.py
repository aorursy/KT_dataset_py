
{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14966,
      "digest": "sha256:36435067423ac8c71d6861deb476e225c4ac9b140c85618be38ca854c4e9a4b1"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112943238,
         "digest": "sha256:6db6199fec8df3d06191df55be7898c3e1a0b8389371dbb86591c6710e0429cb"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 425,
         "digest": "sha256:b89ff65d69ce89fe9d05fe3acf9f89046a19eaed148e80a6e167b93e6dc26423"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5476,
         "digest": "sha256:d7a15e9b63f265b3f895e4c9f02533d105d9b277e411b93e81bb98972018d11a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1952,
         "digest": "sha256:f40e3a2f47878ee1eae7a6b962bff3f7bb2c47baceacc04c3eb29412bb981298"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2483992161,
         "digest": "sha256:3ed1085f84930ef13f3e8c18f9d13d41d88d325d8d6f846112253f61243aaf8c"
      }
   ]
}

import h5py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import scipy as sp
import random
import nilearn as nl
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
import category_encoders as ce
from sklearn.metrics import mean_squared_error

import torch

import lightgbm as lgb
from glob import glob

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv').sort_values(by='Id')

loadings = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')

fnc = pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')

sample = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')

reveal = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv')

ICN = pd.read_csv('/kaggle/input/trends-assessment-prediction/ICN_numbers.csv')


!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
mat = h5py.File('/kaggle/input/trends-assessment-prediction/fMRI_train/10031.mat','r')
mat.keys()
sample = mat['SM_feature']

array = sample[()]
array.shape
print(array.min(),array.max(),array.mean())
mat, ax = plt.subplots(1,4)
mat.set_size_inches(25, 10)
for i in range(4):
    Temp = array[i*10, :, 10, :] !=0  
    ax[i].imshow(Temp)
plt.show()
motor_images = datasets.fetch_neurovault_motor_task()
img = motor_images.images[0]
nii_loc = "/kaggle/input/trends-assessment-prediction/fMRI_mask.nii"
nii_loc2 = "/kaggle/input/trends-assessment-prediction/fMRI_train/10031.mat"
niiplot = plotting.plot_glass_brain(img)
niiplot

maskni = nl.image.load_img(nii_loc)
subjectimage = nl.image.new_img_like(nii_loc, array, affine=maskni.affine, copy_header=True)

smri = 'ch2better.nii'
num_components = subjectimage.shape[-1]
grid_size = int(np.ceil(np.sqrt(num_components)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subjectimage)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img, bg_img=smri, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)
train.isnull().sum()
reveal.head
ICN.head()
fnc.head()
train.head()
train_ids = sorted(loadings[loadings['Id'].isin(train.Id)]['Id'].values)
test_ids = sorted(loadings[~loadings['Id'].isin(train.Id)]['Id'].values)
predictions = pd.DataFrame(test_ids, columns=['Id'], dtype=str)
features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
data = pd.merge(loadings, train, on='Id').dropna()
X_train = data.drop(list(features), axis=1).drop('Id', axis=1)
y_train = data[list(features)]
X_test = loadings[loadings.Id.isin(test_ids)].drop('Id', axis=1)
model = RandomForestRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5
)
cv = KFold(n_splits = 5, shuffle=True, random_state=29)
grid = {
    'n_estimators':[5,10,20,100]
}
gs = GridSearchCV(model, grid, n_jobs=-1, cv=cv, verbose=1, scoring='neg_mean_absolute_error')
best_models = {}
for col in features:
    gs.fit(X_train, y_train[col])
    best_models[col] = gs.best_estimator_
    print(gs.best_score_)
for col in features:
    predictions[col] = best_models[col].predict(X_test)
def make_sub(predictions):
    features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
    _columns = (0,1,2,3,4)
    tests = predictions.rename(columns=dict(zip(features, _columns)))
    tests = tests.melt(id_vars='Id',value_vars=_columns,value_name='Predicted')
    tests['target'] = tests.variable.map(dict(zip(_columns, features)))
    tests['Id_'] = tests[['Id', 'target']].apply(lambda x: '_'.join((str(x[0]), str(x[1]))), axis=1)
  
    return tests.sort_values(by=['Id', 'variable'])\
              .drop(['Id', 'variable', 'target'],axis=1)\
              .rename(columns={'Id_':'Id'})\
              .reset_index(drop=True)\
              [['Id', 'Predicted']]
sub = make_sub(predictions)
sub.head()
sub.to_csv('firsttry.csv', index=False)