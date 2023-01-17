import os

import h5py

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
os.listdir('/kaggle/input/trends-assessment-prediction/')
import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
def metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
import cudf

from cuml import SVR
fnc_df = cudf.read_csv("../input/trends-assessment-prediction/fnc.csv")

loading_df = cudf.read_csv("../input/trends-assessment-prediction/loading.csv")





fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")





labels_df = cudf.read_csv("../input/trends-assessment-prediction/train_scores.csv")

labels_df["is_train"] = True



df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
loading_df.shape
temp_data =  train_data.drop(['Id'], axis=1)

plt.figure(figsize = (15, 10))

sns.heatmap(temp_data.corr(), annot = True, cmap="brg")

plt.yticks(rotation=0) 

plt.show()
fnc_df.head()
labels_df.head()
labels_df.isnull().sum()

temp_data =  loading_df.drop(['Id'], axis=1)

plt.figure(figsize = (20, 20))

sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")

plt.yticks(rotation=0) 

plt.show()
import nilearn as nl

import nibabel as nib

from nilearn import image

from nilearn import plotting

from nilearn import datasets

from nilearn import surface

import nilearn.plotting as nlplt
fmri_mask = '../input/trends-assessment-prediction/fMRI_mask.nii'
smri = 'ch2better.nii'

mask_img = nl.image.load_img(fmri_mask)



def load_subject(filename, mask_img):

    subject_data = None

    with h5py.File(filename, 'r') as f:

        subject_data = f['SM_feature'][()]

    # It's necessary to reorient the axes, since h5py flips axis order

    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])

    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)



    return subject_img





files = random.choices(os.listdir('../input/trends-assessment-prediction/fMRI_train/'), k = 3)

for file in files:

    subject = os.path.join('../input/trends-assessment-prediction/fMRI_train/', file)

    subject_img = load_subject(subject, mask_img)

    print("Image shape is %s" % (str(subject_img.shape)))

    num_components = subject_img.shape[-1]

    print("Detected {num_components} spatial maps".format(num_components=num_components))

    rsn = subject_img

    #convert to 3d image

    first_rsn = image.index_img(rsn, 0)

    print(first_rsn.shape)     

    plotting.plot_glass_brain(first_rsn,display_mode='lyrz')

    print("-"*50)
motor_images = datasets.fetch_neurovault_motor_task()

stat_img = motor_images.images[0]

view = plotting.view_img_on_surf(stat_img, threshold='90%')

view.open_in_browser()

view
FNC_SCALE = 1/500



df[fnc_features] *= FNC_SCALE

test_df[fnc_features] *= FNC_SCALE
%%time



NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)





features = loading_features + fnc_features



overal_score = 0

for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    

    y_oof = np.zeros(df.shape[0])

    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))

    

    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):

        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

        train_df = train_df[train_df[target].notnull()]



        model = SVR(C=c, cache_size=3000.0)

        model.fit(train_df[features], train_df[target])



        y_oof[val_ind] = model.predict(val_df[features])

        y_test[:, f] = model.predict(test_df[features])

        

    df["pred_{}".format(target)] = y_oof

    test_df[target] = y_test.mean(axis=1)

    

    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)

    overal_score += w*score

    print(target, np.round(score, 4))

    print()

    

print("Overal score:", np.round(overal_score, 4))
sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.head(10)
sub_df.to_csv("submission.csv", index=False)