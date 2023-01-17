import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
import warnings;
warnings.filterwarnings('ignore');

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.metrics import accuracy_score
train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
train.target.value_counts()
train['sex'] = train['sex'].fillna('na')
train['age_approx'] = train['age_approx'].fillna(0)
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

test['sex'] = test['sex'].fillna('na')
test['age_approx'] = test['age_approx'].fillna(0)
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train['sex'] = train['sex'].astype("category").cat.codes +1
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1
train.head()
test['sex'] = test['sex'].astype("category").cat.codes +1
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1
test.head()
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]
y_train = train['target']

x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]

train_DMatrix = xgb.DMatrix(x_train, label= y_train)
test_DMatrix = xgb.DMatrix(x_test)

param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 5
}
epochs = 100
clf = xgb.XGBClassifier(n_estimators=1000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (32542/584))
clf.fit(x_train, y_train)
sub.head()
sub["meta_target"] = clf.predict_proba(x_test)[:,1]
#sub.to_csv('submission.csv', index = False)
from fastai.imports import *
from fastai import *
from fastai.vision import *
from torchvision.models import *
train.head()
train.shape
label1 = train[train['target']==1]
label1.shape
label2 = train[train['target']==0].iloc[:584]
label2.head()
train_small = pd.concat([label1,label2])
train_small.head()
train_small.shape
train_small['image_name'] = train_small['image_name']+'.jpg'
test['image_name'] = test['image_name']+'.jpg'
train_small.head()
train_small.to_csv('train_jpg.csv', index = False)
test.to_csv('test_jpg.csv', index = False)
tfms = get_transforms(flip_vert=True)
path = "/kaggle/"
data = ImageDataBunch.from_csv(path, folder= 'input/siim-isic-melanoma-classification/jpeg/train', 
                              valid_pct = 0.2,
                              csv_labels = 'working/train_jpg.csv',
                              ds_tfms = tfms, 
                              fn_col = 'image_name',
                              label_col = 'target',
                              bs = 32,
                              size=256).normalize(imagenet_stats);
test_data = ImageDataBunch.from_csv(path, folder= 'input/siim-isic-melanoma-classification/jpeg/test', 
                              valid_pct = 0.2,
                              csv_labels = 'working/test_jpg.csv',
                              ds_tfms = tfms, 
                              fn_col = 'image_name',
                              bs = 32,
                              size=256).normalize(imagenet_stats);

data.show_batch(rows=3,figsize=(8,8));
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learn.model.to(device)
#learn.model_dir = "/kaggle/working"
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8,slice(0.015));
learn.freeze()