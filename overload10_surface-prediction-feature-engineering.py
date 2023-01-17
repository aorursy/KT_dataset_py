# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.tabular import *

import seaborn as sns

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df_X = pd.read_csv('../input/X_train.csv',sep=',')

train_df_Y = pd.read_csv('../input/y_train.csv')

test_df_X = pd.read_csv('../input/X_test.csv')



train_df_X.shape, train_df_Y.shape, test_df_X.shape
train_df_X.isnull().sum()
train_df_X.head()
train_df_Y.head()
test_df_X.head()
train_df_Y.surface.value_counts()
train_df_X.isnull().sum()
train_df_X.shape[0]/128, train_df_Y.shape,test_df_X.shape[0]/128,
train_df_X.shape
train_df_X.describe()
from scipy.stats import kurtosis

from scipy.stats import skew



def _kurtosis(x):

    return kurtosis(x)



def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))



def quaternion_to_euler(x, y, z, w):

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
train_df_X.columns
def create_Feature(org_df):

    df = org_df.copy()

    df['norm_quat'] = (df['orientation_X']**2 + df['orientation_Y']**2 + df['orientation_Z']**2 + df['orientation_W']**2)

    df['mod_quat'] = (df['norm_quat'])**0.5

    df['norm_X'] = df['orientation_X'] / df['mod_quat']

    df['norm_Y'] = df['orientation_Y'] / df['mod_quat']

    df['norm_Z'] = df['orientation_Z'] / df['mod_quat']

    df['norm_W'] = df['orientation_W'] / df['mod_quat']

    

    df['total_angular_velocity'] = (df['angular_velocity_X'] ** 2 + df['angular_velocity_Y'] ** 2 + df['angular_velocity_Z'] ** 2) ** 0.5

    df['total_linear_acceleration'] = (df['linear_acceleration_X'] ** 2 + df['linear_acceleration_Y'] ** 2 + df['linear_acceleration_Z'] ** 2) ** 0.5

    df['acc_vs_vel'] = df['total_linear_acceleration'] / df['total_angular_velocity']

    

    x, y, z, w = df['orientation_X'].tolist(), df['orientation_Y'].tolist(), df['orientation_Z'].tolist(), df['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    df['euler_x'] = nx

    df['euler_y'] = ny

    df['euler_z'] = nz

    

    df['total_angle'] = (df['euler_x'] ** 2 + df['euler_y'] ** 2 + df['euler_z'] ** 2) ** 5

    df['angle_vs_acc'] = df['total_angle'] / df['total_linear_acceleration']

    df['angle_vs_vel'] = df['total_angle'] / df['total_angular_velocity']

    

    ndf=pd.DataFrame()

    df['angular_velocity'] = df['angular_velocity_X'] + df['angular_velocity_Y'] + df['angular_velocity_Z']

    df['linear_acceleration'] = df['linear_acceleration_X'] + df['linear_acceleration_Y'] + df['linear_acceleration_Z']

    df['velocity_to_acceleration'] = df['angular_velocity'] / df['linear_acceleration']

    for c in df.columns[3:]:

        ndf[c+'_min']=df.groupby(['series_id'])[c].min()

        ndf[c+'_max']=df.groupby(['series_id'])[c].max()        

        ndf[c+'_absMax'] =df.groupby(['series_id'])[c].apply(lambda x : np.max(np.abs(x)))

        ndf[c+'_absMin'] =df.groupby(['series_id'])[c].apply(lambda x : np.min(np.abs(x)))

        ndf[c+'_mean']=df.groupby(['series_id'])[c].mean()

        ndf[c+'_mean']=df.groupby(['series_id'])[c].median()

        ndf[c+'_std']=df.groupby(['series_id'])[c].std()

#         ndf[c+'_log']=df.groupby(['series_id'])[c].apply(lambda x : np.log(x))

#         ndf[c+'_maxtomin']=df[c + '_max'] / df[c + '_min']

        ndf[c + '_q25'] = df.groupby(['series_id'])[c].quantile(0.25)

        ndf[c + '_q50'] = df.groupby(['series_id'])[c].quantile(0.5)

        ndf[c + '_q75'] = df.groupby(['series_id'])[c].quantile(0.75)

#         ndf[c + '_CPT5'] = df.groupby(['series_id'])[c].apply(CPT5) 

        ndf[c + '_SSC'] = df.groupby(['series_id'])[c].apply(SSC) 

        ndf[c + '_skewness'] = df.groupby(['series_id'])[c].apply(skewness)

        ndf[c + '_wave_lenght'] = df.groupby(['series_id'])[c].apply(wave_length)

        ndf[c + '_norm_entropy'] = df.groupby(['series_id'])[c].apply(norm_entropy)

        ndf[c + '_SRAV'] = df.groupby(['series_id'])[c].apply(SRAV)

        ndf[c + '_kurtosis'] = df.groupby(['series_id'])[c].apply(_kurtosis) 

        ndf[c + '_mean_abs'] = df.groupby(['series_id'])[c].apply(mean_abs) 

        ndf[c + '_zero_crossing'] = df.groupby(['series_id'])[c].apply(zero_crossing) 

    return ndf.reset_index()
%time train_fe_X = create_Feature(train_df_X)

%time test_fe_X = create_Feature(test_df_X)

train_fe_X.shape, test_fe_X.shape
train_fe_X.to_feather('Train_FE')

test_fe_X.to_feather('Test_FE')

train_df_Y.to_feather('Target')

# train_df_X.isnull().sum()
train_fe_X.shape
# x.loc[x['series_id']==4]
# train_df = pd.merge(train_fe_X,train_df_Y,how='inner',on='series_id')

# train_df.shape
# train_df.isnull().sum()
# test_fe_X.shape
train_fe_X.to_feather('Train_FE')

test_fe_X.to_feather('Test_FE')

train_df_Y.to_feather('Target')

# train_df_X.isnull().sum()
# seed=22

# train_samples = train_df.sample(frac=0.8, random_state=seed)

# valid_samples = train_df.drop(train_samples.index)

# train_samples.loc[:,'isValid'] = 0

# valid_samples.loc[:,'isValid'] = 1
# train_samples.shape,valid_samples.shape
# valid_idx = valid_samples.index
# final_train = pd.concat([train_samples,valid_samples]).reset_index(drop=True)

# final_train.shape
# Creating upsample for smaller category of data



# undersampled_hard_tiles_train = train_samples.loc[(train_samples.surface == 'hard_tiles'),:]

# undersampled_carpet_train = train_samples.loc[(train_samples.surface == 'carpet'),:]

# dominant_class_train = train_samples.loc[ (train_samples.surface != 'hard_tiles') & (train_samples.surface != 'carpet'),:]





# undersampled_hard_tiles_train.shape, undersampled_carpet_train.shape, dominant_class_train.shape



# upsample_hard_tile = pd.concat([undersampled_hard_tiles_train for i in range(0,18)]).reset_index(drop=True)

# upsample_carpet = pd.concat([undersampled_carpet_train for i in range(0,2)]).reset_index(drop=True)

# upsample_hard_tile.shape , upsample_carpet.shape



# Augmenting data by using random noise



# final_train = pd.concat([upsample_hard_tile,upsample_carpet,dominant_class_train,valid_samples]).reset_index(drop=True)

# final_train.loc[:,'randomnoise'] = [random.random() for i in range(0,final_train.copy().shape[0])]

# final_train.shape
# final_train.head()
# final_train.surface.value_counts()
# test_fe_X =  pd.read_feather('Test_FE')

# test_fe_X.shape
# test_final_X = test_fe_X.copy()

# test_final_X.loc[:,'randomnoise'] = [random.random() for i in range(0,test_fe_X.copy().shape[0])]

# test_final_X.shape
# test_final_X = test_fe_X.copy()

# test_final_X.shape
# final_train.columns

# con_var = final_train.columns

# con_var=con_var.drop(labels=['series_id','group_id','surface','isValid'])

# print(con_var)

# dep_var ='surface'

# procs = [FillMissing, Categorify, Normalize]
# data = (TabularList.from_df(final_train, cont_names=con_var, procs=procs)

#                             .split_by_idx(valid_idx=valid_idx)

#                             .label_from_df(cols=dep_var)

#                             .add_test(TabularList.from_df(test_final_X, cont_names=con_var, procs=procs))

#                             .databunch())
# data.show_batch(rows=10,ds_type=DatasetType.Valid)
# len(data.train_ds),len(data.valid_ds), len(data.test_ds)
# learn = tabular_learner(data, layers=[400,200], metrics=accuracy)
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(18, 3e-03)
# learn.recorder.plot_losses()
# learn.save('stage-1')
# learn.load('stage-1')
# learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(10, 3e-05)
# learn.recorder.plot_losses()
# learn.save('stage-2')
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()
# losses,idxs = interp.top_losses()

# len(data.valid_ds)==len(losses)==len(idxs)
# interp.most_confused()
# learn.load('stage-1')

# interp = ClassificationInterpretation.from_learner(learn)

# interp.plot_confusion_matrix()
# interp.most_confused()
# preds = np.argmax(learn.get_preds(ds_type=DatasetType.Test)[0],axis=1)
# pred_label=[]

# for row in preds:

#     pred_label.append(data.classes[row])

# # pred_label
# len(pred_label)
# submission = pd.DataFrame({"series_id": test_final_X.series_id, "surface": pred_label})

# submission.to_csv("submission_DL.csv", index = False)

# submission.head(10)