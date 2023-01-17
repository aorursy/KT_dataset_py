import numpy as np

import pandas as pd

from PIL import Image

import os

directory_train = "../input/train/train/"

directory_test = "../input/test/test/"
train_labels = pd.read_csv("../input/train.csv")

train_labels.head()
image_names = []

for filename in os.listdir(directory_train):

    image_names.append(filename)

image_names.sort()
test_image_names = []

for filename in os.listdir(directory_test):

    test_image_names.append(filename)

test_image_names.sort()
images = []

for filename in image_names:

    im = Image.open(directory_train+filename, 'r')

    pix_val = list(im.getdata())

    pix_val_flat = [x for sets in pix_val for x in sets]

    images.append(pix_val_flat)
test_images = []

for filename in test_image_names:

    im = Image.open(directory_test+filename, 'r')

    pix_val = list(im.getdata())

    pix_val_flat = [x for sets in pix_val for x in sets]

    test_images.append(pix_val_flat)
col_names = ["pxl_"+str(i) for i in range(len(images[0]))]



df_train=pd.DataFrame(images,columns=col_names)

df_train["has_cactus"] = train_labels.has_cactus 



df_test = pd.DataFrame(test_images,columns=col_names)
df_train.head()
df_test.head()
#train_size = 5000

df = df_train#.iloc[:train_size]
"""

!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm

!git clone --recursive https://github.com/Microsoft/LightGBM

    

!apt-get install -y -qq libboost-all-dev



%%bash

cd LightGBM

rm -r build

mkdir build

cd build

cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

make -j$(nproc)



!cd LightGBM/python-package/;python3 setup.py install --precompile



!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

!rm -r LightGBM

!nvidia-smi

"""
params = {'num_leaves': 15,

         'min_data_in_leaf': 50,

         'objective': 'binary',

         'max_depth': 25,

         'learning_rate': 0.01,#0.0123,

         'boosting': 'goss',

         'feature_fraction': 0.7,

         'reg_alpha': 1.728,

         'reg_lambda': 4.984,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01,

         'min_child_weight': 19.4,

         'num_threads': 4,

        # 'device': 'gpu',

        #'gpu_platform_id': 0,

        #'gpu_device_id': 0

         }
import time

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from sklearn import metrics



for i in range(4):

    print(len(df))

    t1=time.time()

    target = 'has_cactus'

    predictors = df.columns.values.tolist()[:-1]

    nfold=5

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

    oof = np.zeros(len(df))

    predictions = np.zeros(len(df_test))



    i = 1

    for train_index, valid_index in skf.split(df, df[target].values):

        print("\nfold {}".format(i))

        xg_train = lgb.Dataset(df.iloc[train_index][predictors].values,

                               label=df.iloc[train_index][target].values,

                               feature_name=predictors,

                               free_raw_data = False

                               )

        xg_valid = lgb.Dataset(df.iloc[valid_index][predictors].values,

                               label=df.iloc[valid_index][target].values,

                               feature_name=predictors,

                               free_raw_data = False

                               )   

        clf = lgb.train(params, xg_train, 2000000, valid_sets = [xg_valid], verbose_eval=1000, early_stopping_rounds = 2000)

        oof[valid_index] = clf.predict(df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 



        predictions += clf.predict(df_test[predictors], num_iteration=clf.best_iteration) / nfold

        i = i + 1

    t2=time.time()

    print((t2-t1)/60)

    print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(df[target].values, oof)))

    df_test["has_cactus"] = predictions

    df = pd.concat([df, df_test[df_test.has_cactus > 0.95]])

    df = pd.concat([df, df_test[df_test.has_cactus < 0.05]])

    df["has_cactus"] = df["has_cactus"].round(0).astype(int)

    df_test = df_test.drop(columns=["has_cactus"])
"""

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot



auc = roc_auc_score(df[target].values, oof)

print('AUC: %.3f' % auc)

fpr, tpr, thresholds = roc_curve(df[target].values, oof)

pyplot.plot([0, 1], [0, 1], linestyle='--')

pyplot.plot(fpr, tpr, marker='.')

pyplot.show()

"""
#predictions.tolist()
sub = pd.read_csv("../input/sample_submission.csv")

sub.has_cactus = predictions

sub.to_csv("submission_lgbm_4runs_5thres.csv",index=False)