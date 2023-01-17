!pip install efficientnet_pytorch
!mkdir /kaggle/working/alias

!ln -s /kaggle/input/data/images_001/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_002/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_003/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_004/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_005/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_006/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_007/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_008/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_009/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_010/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_011/images/*.png /kaggle/working/alias

!ln -s /kaggle/input/data/images_012/images/*.png /kaggle/working/alias
import os

import pickle

import platform

import shutil

import sys

import warnings



from pathlib import Path



import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pydicom

import scipy.ndimage

import sklearn.metrics



from fastai.callbacks import *

from fastai.callbacks.hooks import *

from fastai.distributed import *

from fastai.vision import *

from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score

from sklearn.model_selection import KFold

from tqdm import tqdm



from efficientnet_pytorch import EfficientNet
path_working_dir = Path().resolve()

path_input_nih   = (path_working_dir / f'../input/data').resolve()

path_input_alias = (path_working_dir / f'./alias').resolve()

path_input_model = (path_working_dir / f'../input/nih-chest-xrays-trained-models').resolve()
IMAGE_SIZE  = 224

BATCH_SIZE  = 24
tfms  = get_transforms(do_flip=False, flip_vert=False, max_rotate=20, max_zoom=1.2, max_warp=0.25, p_affine=0.7, max_lighting=0.4, p_lighting=0.5)
df_input_all                   = pd.read_csv(path_input_nih / 'Data_Entry_2017.csv')

df_input_all['Finding Labels'] = df_input_all['Finding Labels'].str.replace('No Finding', '')

df_list_train                  = pd.read_csv(path_input_nih / 'train_val_list.txt', header=None)

df_list_valid                  = pd.read_csv(path_input_nih / 'test_list.txt', header=None)



list_all       = df_input_all['Image Index'].tolist()

list_train     = df_list_train[0].tolist()

list_valid     = df_list_valid[0].tolist()

list_idx_train = [True if fname in list_train else False for fname in tqdm(list_all)]

list_idx_valid = [True if fname in list_valid else False for fname in tqdm(list_all)]

list_classes   = sorted([target for target in set(df_input_all['Finding Labels'].tolist()) if not '|' in target and target != ''])



df_input_train = df_input_all[list_idx_train].reset_index(drop=True)[['Image Index', 'Finding Labels']]

df_input_valid = df_input_all[list_idx_valid].reset_index(drop=True)[['Image Index', 'Finding Labels']]

df_input_merge = pd.concat([df_input_train, df_input_valid]).reset_index(drop=True)
img_list         = ImageList.from_df(df_input_merge, path_input_alias, convert_mode='L')

np_y_value_valid = LongTensor(np.array([[1 if list_classes[i] in label else 0 for label in df_input_valid['Finding Labels'].tolist()] for i in range(14)]).T)
list_BATCH_SIZE       = [160, 112, 104, 80, 56, 40, 32, 24]

list_np_H_value_valid = []



for x in range(8):

    data = (img_list.split_by_idxs(list(range(len(df_input_train))), list(range(len(df_input_train), len(df_input_train)+len(df_input_valid))))

                    .label_from_df(cols='Finding Labels', classes=list_classes, label_delim='|')

                    .transform(tfms, size=IMAGE_SIZE)

                    .databunch(bs=list_BATCH_SIZE[x], num_workers=os.cpu_count()))

    bx    = f'b{x}'

    model = EfficientNet.from_pretrained(f'efficientnet-{bx}', num_classes=14, in_channels=1)

    learn = Learner(data, model)

    learn = learn.load(path_input_model / f'efficientnet-b{x}_224x224x1_epoch_30/model_unfreeze_best')

    np_H_value_valid, np_H_01_valid, np_loss_valid = learn.get_preds(DatasetType.Valid, with_loss=True)

    list_np_H_value_valid.append(np_H_value_valid)

    np.save(f'np_H_value_valid_b{x}.npy', np.array(np_H_value_valid))

    # np_H_value_tta_beta_valid, np_H_value_tta_avg_valid, np_loss_tta_valid = learn.TTA(ds_type=DatasetType.Valid, with_loss=True, beta=None)

    # np.save(f'np_H_value_tta_beta_valid_b{x}.npy', np.array(np_H_value_tta_beta_valid))

    # np.save(f'np_H_value_tta_avg_valid_b{x}.npy',  np.array(np_H_value_tta_avg_valid))
list_output_auc = []

for x in range(8):

    list_output_auc.append([float(auc_roc_score(list_np_H_value_valid[x][:, i], np_y_value_valid[:, i])) for i in range(14)])

df_output_auc = pd.DataFrame(list_output_auc, columns=list_classes)

df_output_auc.index = [f'b{x}' for x in range(8)]

df_output_auc
list_output_accuracy = []

for x in range(8):

    list_output_accuracy.append([float(accuracy_score(np_y_value_valid[:, i],list_np_H_value_valid[x][:, i] >= 0.5)) for i in range(14)])

df_output_accuracy = pd.DataFrame(list_output_accuracy, columns=list_classes)

df_output_accuracy.index = [f'b{x}' for x in range(8)]

df_output_accuracy
list_output_logloss = []

for x in range(8):

    list_output_logloss.append([float(log_loss(np_y_value_valid[:, i], list_np_H_value_valid[x][:, i])) for i in range(14)])

df_output_logloss = pd.DataFrame(list_output_logloss, columns=list_classes)

df_output_logloss.index = [f'b{x}' for x in range(8)]

df_output_logloss
list_output_tp = []

for x in range(8):

    list_output_tp.append([int(confusion_matrix(np_y_value_valid[:, i],list_np_H_value_valid[x][:, i] >= 0.5).ravel()[3]) for i in range(14)])

df_output_tp = pd.DataFrame(list_output_tp, columns=list_classes)

df_output_tp.index = [f'b{x}' for x in range(8)]

df_output_tp
list_output_fp = []

for x in range(8):

    list_output_fp.append([int(confusion_matrix(np_y_value_valid[:, i],list_np_H_value_valid[x][:, i] >= 0.5).ravel()[1]) for i in range(14)])

df_output_fp = pd.DataFrame(list_output_fp, columns=list_classes)

df_output_fp.index = [f'b{x}' for x in range(8)]

df_output_fp
list_output_fn = []

for x in range(8):

    list_output_fn.append([int(confusion_matrix(np_y_value_valid[:, i],list_np_H_value_valid[x][:, i] >= 0.5).ravel()[2]) for i in range(14)])

df_output_fn = pd.DataFrame(list_output_fn, columns=list_classes)

df_output_fn.index = [f'b{x}' for x in range(8)]

df_output_fn
list_output_tn = []

for x in range(8):

    list_output_tn.append([int(confusion_matrix(np_y_value_valid[:, i],list_np_H_value_valid[x][:, i] >= 0.5).ravel()[0]) for i in range(14)])

df_output_tn = pd.DataFrame(list_output_tn, columns=list_classes)

df_output_tn.index = [f'b{x}' for x in range(8)]

df_output_tn