import numpy as np
import cv2
import tensorflow as tf
import keras
import torch
import matplotlib
!nvcc --version
!cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
print('LIBRARY VERSIONS')
print('numpy', np.__version__)
print('cv2', cv2.__version__)
print('matplotlib', matplotlib.__version__)
print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('torch', torch.__version__)
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
# from tqdm import tqdm

# pd.options.display.max_rows = 999
# pd.options.display.max_columns = 999
import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li
print(os.listdir('../input/train/')[:5])
print(os.listdir('../input/test/')[:5])
train_ids = get_path('*.jpg', False, '../input/train/')
print(len(train_ids))
print(train_ids[:5])
test_ids = get_path('*.jpg', False, '../input/test/')
print(len(test_ids))
print(test_ids[:5])
def get_id(path):
    return path.split('/')[-1].split('.jpg')[0]
    
def get_target(x):
    if 'cat' in x:
        return 1
    elif 'dog' in x:
        return 0
    else:
        return -1

evals = {
    'path': train_ids+test_ids,
    'is_test': [0]*len(train_ids)+[1]*len(test_ids),
}
evals = pd.DataFrame(evals)
evals['img_id'] = evals['path'].apply(get_id)
evals['target'] = evals['img_id'].apply(get_target)
evals['eval_set'] = -1
cols = ['img_id', 'target', 'path', 'is_test', 'eval_set']
evals = evals[cols]
from sklearn.model_selection import KFold
train_num = len(train_ids)
n_splits = 5
random_state = 42
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
for fold_i, (_,val_idx) in enumerate(kf.split(np.arange(train_num))):
    evals.loc[val_idx, 'eval_set'] = fold_i
evals['eval_set'][:train_num].hist(bins=9)
pd.concat([evals.head(), evals.tail()])
evals.to_csv('evals.csv', index=False)

