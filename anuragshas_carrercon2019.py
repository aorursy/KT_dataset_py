!pip install fastai==1.0.47
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from pathlib import Path

# Any results you write to the current directory are saved as output.

PATH = Path('../input')
import fastai

from fastai.tabular import *

import tqdm
!head -n 5 ../input/X_train.csv
features = ['orientation_X',

'orientation_Y',

'orientation_Z',

'orientation_W',

'angular_velocity_X',

'angular_velocity_Y',

'angular_velocity_Z',

'linear_acceleration_X',

'linear_acceleration_Y',

'linear_acceleration_Z']
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z
def augment_df(actual):

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    return actual
augment_features = [

    'total_angular_velocity',

    'total_linear_acceleration',

    'acc_vs_vel',

    'euler_x',

    'euler_y',

    'euler_z',

    'total_angle',

    'angle_vs_acc',

    'angle_vs_vel'

]
total_features = features+augment_features
def gen_features(df):

    strain = []

    for feature in total_features:

        X = df[feature]

        abs_diff = np.abs(np.diff(X))

        min_x = X.min()

        max_x = X.max()

        strain.append(X.mean()) #0

        strain.append(X.std()) #1

        strain.append(min_x) #2

        strain.append(max_x) #3

        strain.append(X.kurtosis()) #4

        strain.append(X.skew()) #5

        strain.append(np.quantile(X,0.01)) #6

        strain.append(np.quantile(X,0.05)) #7

        strain.append(np.quantile(X,0.95)) #10

        strain.append(np.quantile(X,0.99)) #11

        strain.append(np.abs(X).max()) #12

        strain.append(np.abs(X).mean()) #13

        strain.append(np.abs(X).std()) #14

        strain.append(np.square(X).kurtosis()) #15

        strain.append(X.mad()) #16

        strain.append(np.mean(np.diff(abs_diff)))#17

        strain.append(np.mean(abs_diff))#18

        strain.append(max_x/min_x)#19

    return pd.Series(strain)
x_train_dfs = pd.read_csv(PATH/'X_train.csv',index_col='row_id',chunksize=128,iterator=True)

x_test_dfs = pd.read_csv(PATH/'X_test.csv',index_col='row_id',chunksize=128,iterator=True)
y_train_df = pd.read_csv(PATH/'y_train.csv')

y_train_df.head()
X_train = pd.DataFrame()

y_train = pd.Series()

for df in tqdm.tqdm_notebook(x_train_dfs):

    ch = augment_df(df)

    ch = gen_features(ch)

    X_train = X_train.append(ch, ignore_index=True)
train_data = X_train

train_data['surface'] = y_train_df['surface']

del(X_train)
train_data.info()
train_data.head()
X_test = pd.DataFrame()

for df in tqdm.tqdm_notebook(x_test_dfs):

    ch = augment_df(df)

    ch = gen_features(ch)

    X_test = X_test.append(ch, ignore_index=True)
procs = [Normalize]

cont_vars = [x for x in range(train_data.shape[1]-1)]


data = (TabularList.from_df(train_data,procs=procs,cont_names=cont_vars)

                .split_by_rand_pct(0.01)

                .label_from_df(cols='surface')

                .add_test(TabularList.from_df(X_test))

                .databunch())
len(data.train_ds.cont_names)
data.classes,data.c
learn = tabular_learner(data, layers=[1024,512], ps=[0.001,0.01], emb_drop=0.1,metrics=[accuracy])
learn.model
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(15, 1e-3, wd=0.2)
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,1e-4,wd=0.2)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
test_preds=learn.get_preds(ds_type=DatasetType.Test)
test_preds[0].argmax(1).size()
test_preds_list = test_preds[0].argmax(1)
labelled_preds = [learn.data.classes[pred] for pred in test_preds_list]
sub_csv = pd.read_csv(PATH/'sample_submission.csv')
sub_csv.head()
sub_csv['surface'] = labelled_preds
sub_csv.to_csv('results.csv',index=False)