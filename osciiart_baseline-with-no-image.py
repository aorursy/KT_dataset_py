import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os, glob, pickle, time, gc, copy, sys

import warnings

import requests



import pydicom

import cv2

import os, os.path as osp



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
# load train data

df_train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

print(df_train.shape)

df_train.head()
# extract exam (study) level data

col_index = 'SOPInstanceUID'

col_groupby = 'StudyInstanceUID'

df_train_study = df_train[df_train[col_groupby].duplicated()==False].reset_index(drop=True)

print(df_train_study.shape)

df_train_study.head()
# calculate q_i

df_tmp = df_train.groupby(col_groupby)['pe_present_on_image'].agg(len).reset_index()

df_tmp.columns = [col_groupby, 'num_images']

df_train_study = pd.merge(df_train_study, df_tmp, on=col_groupby, how='left')



df_tmp = df_train.groupby(col_groupby)['pe_present_on_image'].agg('sum').reset_index()

df_tmp.columns = [col_groupby, 'm_i']

df_train_study = pd.merge(df_train_study, df_tmp, on=col_groupby, how='left')



df_train_study['q_i'] = df_train_study['m_i'] /df_train_study['num_images'] 

df_train = pd.merge(df_train, df_train_study[[col_groupby, 'num_images', 'q_i']], on=col_groupby, how='left')

df_train.head()
# calculate average

col_index = 'SOPInstanceUID'

col_targets = [

    'negative_exam_for_pe',

    'indeterminate',

    'chronic_pe',

    'acute_and_chronic_pe',

    'central_pe',

    'leftsided_pe',

    'rightsided_pe',

    'rv_lv_ratio_gte_1',

    'rv_lv_ratio_lt_1',

    'pe_present_on_image',

]

mean_targets = np.zeros(len(col_targets), np.float32)

for i, col in enumerate(col_targets[:-1]):

    mean_targets[i] = df_train_study[col].mean()

mean_targets[-1] = df_train[col_targets[-1]].mean()

preds_mean_study = np.ones([len(df_train_study), len(col_targets[:-1])], np.float32) * mean_targets[:-1][np.newaxis]

preds_mean_image = np.ones(len(df_train), np.float32) * mean_targets[-1]



for i, col in enumerate(col_targets):

    print("{} average: {:.6f}".format((col +" "*50)[:30], mean_targets[i]))
# calculate metrics

from sklearn import metrics

def calc_metrics(y_true_exam, y_pred_exam, y_true_imag, y_pred_imag, q_image):

    weights = np.array([

        0.0736196319, 

        0.09202453988, 

        0.1042944785, 

        0.1042944785, 

        0.1877300613, 

        0.06257668712, 

        0.06257668712,

        0.2346625767,

        0.0782208589,

        0.07361963,

    ])

    score_list = []

    scores = {}

    for i in range(9):

        bce = metrics.log_loss(y_true_exam[:,i], y_pred_exam[:,i])

        scores[col_targets[i]] = bce

        score_list.append(bce)

        print("{}: {:.6f}".format((col_targets[i]+" "*50)[:30], bce))

    score_s = np.sum(weights[:-1]*np.array(score_list))

    

    scores["exam_level_weighted_log_loss"] = score_s

    

    print("{}: {:.6f}".format(("exam_level_weighted_log_loss" +" "*50)[:30], score_s))

    score_i =  np.sum(- q_image * (y_true_imag*np.log(y_pred_imag) + (1-y_true_imag)*np.log(1-y_pred_imag))) / np.sum(q_image)

    scores["image_level_weighted_log_loss"] = score_i

    print("{}: {:.6f}".format(("image_level_weighted_log_loss" +" "*50)[:30], score_i))

#     scores.append(score)

    score_all = (score_s * len(y_true_exam) + score_i * np.sum(q_image) * weights[-1]) / (len(y_true_exam)+np.sum(q_image)* weights[-1])

    scores["total_loss"] = score_all

    print("{}: {:.6f}".format(("total_loss" +" "*50)[:30], score_all))

#     print(len(y_true_exam), np.sum(q_image)* weights[-1])

    return scores





_ = calc_metrics(

    y_true_exam=df_train_study[col_targets[:-1]].values, 

    y_pred_exam=preds_mean_study, 

    y_true_imag=df_train[col_targets[-1]].values, 

    y_pred_imag=preds_mean_image, 

    q_image=df_train['q_i'].values)
q_weighted_mean = np.sum(df_train['pe_present_on_image'] * df_train['q_i']) / np.sum(df_train['q_i'])

print("q_weighted_mean: {:.6f}".format(q_weighted_mean))

preds_mean_image_q_weighted = np.ones(len(df_train), np.float32) * q_weighted_mean
_ = calc_metrics(

    y_true_exam=df_train_study[col_targets[:-1]].values, 

    y_pred_exam=preds_mean_study, 

    y_true_imag=df_train[col_targets[-1]].values, 

    y_pred_imag=preds_mean_image_q_weighted, 

    q_image=df_train['q_i'].values)
# get dicom paths

df_train['path'] = ("../input/rsna-str-pulmonary-embolism-detection/train/" 

                   + df_train['StudyInstanceUID'].values + "/"

                   + df_train['SeriesInstanceUID'].values + "/"

                   + df_train['SOPInstanceUID'].values + ".dcm"

                  )

print(df_train['path'][0])
# get series index of image

import multiprocessing

from concurrent.futures import ProcessPoolExecutor



def task(i):

    if (i+1)%100000==0:

        print("{}/{} {:.1f}".format(i+1, len(df_train), time.time()-starttime))

    path = df_train['path'][i]

    tmp_dcm = pydicom.dcmread(path)

    return tmp_dcm.ImagePositionPatient[-1]





starttime = time.time()

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

# futures = [executor.submit(task, i) for i in range(10000)]

futures = [executor.submit(task, i) for i in range(len(df_train.iloc[:]))]

result_list = []

for i in range(len(futures)):

    result_list.append(futures[i].result())

df_train['z_pos'] = 0

df_train['z_pos'][:len(result_list)] = result_list



del futures, result_list

gc.collect()



df_train.head()
# calculate slice location

df_tmp = []

for i in range(len(df_train_study)):

    if (i+1)%1000==0: print("{}/{}".format(i+1, len(df_train_study)))

    study = df_train_study[col_groupby][i]

    df_study = df_train[df_train[col_groupby]==study].sort_values('z_pos').reset_index(drop=True)

    df_study['series_index'] = np.arange(len(df_study))

    df_tmp.append(df_study[[col_index, 'series_index']])

df_tmp = pd.concat(df_tmp)



df_train = pd.merge(df_train, df_tmp, on=col_index, how='left')

# df_test = pd.merge(df_test, df_test_study[[col_groupby, 'num_images']], on=col_groupby, how='left')

df_train['slice_location'] = df_train['series_index'] / (df_train['num_images'] - 1)

df_train.head()
# visualize the relation between pe and slice location

plt.figure(figsize=(10, 5))

plt.hist(df_train['slice_location'][df_train['pe_present_on_image']==True], bins=np.arange(101)/100, label='PE', density=True, alpha=0.3)

plt.hist(df_train['slice_location'][df_train['pe_present_on_image']==False], bins=np.arange(101)/100, label='no PE', density=True, alpha=0.3)

plt.legend()

plt.show()
bins = 8

df_train['bins'] = bins-1

for i in range(bins):

    df_train['bins'][(df_train['slice_location']>=(i/bins)) & (df_train['slice_location']<((i+1)/bins))] = i

df_train.head()
df_train['slice_location'].min(), df_train['slice_location'].max()
for i in range(bins):

    print(i, np.sum(df_train['bins']==i))
q_weighted_means = np.zeros(bins, np.float32)

for i in range(bins):

    tmp_index = df_train['bins']==i

    q_weighted_means[i] = np.sum(df_train['pe_present_on_image'][tmp_index].values * df_train['q_i'][tmp_index].values) / np.sum(df_train['q_i'][tmp_index].values)

df_train['q_weighted_means'] = df_train['bins'].apply(lambda x: q_weighted_means[x])

print(q_weighted_means)

df_train.head()
q_weighted_means = np.array([0.00326324, 0.05970682, 0.32645303, 0.67452216, 0.71344817, 0.4734337, 0.0740926, 0.00369781])

print(q_weighted_means)
_ = calc_metrics(

    y_true_exam=df_train_study[col_targets[:-1]].values, 

    y_pred_exam=preds_mean_study, 

    y_true_imag=df_train[col_targets[-1]].values, 

    y_pred_imag=df_train['q_weighted_means'].values, 

    q_image=df_train['q_i'].values)
# load test data

df_test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

print(df_test.shape)

df_test.head()
# get dicom paths

df_test['path'] = ("../input/rsna-str-pulmonary-embolism-detection/test/" 

                   + df_test['StudyInstanceUID'].values + "/"

                   + df_test['SeriesInstanceUID'].values + "/"

                   + df_test['SOPInstanceUID'].values + ".dcm"

                  )

print(df_test['path'][0])
# extract exam (study) level data

df_test_study = df_test[df_test[col_groupby].duplicated()==False].reset_index(drop=True)

df_tmp = df_test.groupby(col_groupby)[col_index].agg(len).reset_index()

df_tmp.columns = [col_groupby, 'num_images']

df_test_study = pd.merge(df_test_study, df_tmp, on=col_groupby, how='left')

df_test = pd.merge(df_test, df_test_study[[col_groupby, 'num_images']], on=col_groupby, how='left')

print(df_test.shape)

df_test.head()
# get series index of image

def task(i):

    if (i+1)%10000==0:

        print("{}/{} {:.1f}".format(i+1, len(df_test), time.time()-starttime))

    path = df_test['path'][i]

    tmp_dcm = pydicom.dcmread(path)

    return tmp_dcm.ImagePositionPatient[-1]



starttime = time.time()

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

# futures = [executor.submit(task, i) for i in range(10000)]

futures = [executor.submit(task, i) for i in range(len(df_test))]

result_list = []

for i in range(len(futures)):

    result_list.append(futures[i].result())

df_test['z_pos'] = result_list

df_test.head()
# calculate slice location

df_tmp = []

for i in range(len(df_test_study)):

    if (i+1)%100==0: print("{}/{}".format(i+1, len(df_test_study)))

    study = df_test_study[col_groupby][i]

    df_study = df_test[df_test[col_groupby]==study].sort_values('z_pos').reset_index(drop=True)

    df_study['series_index'] = np.arange(len(df_study))

    df_tmp.append(df_study[[col_index, 'series_index']])

df_tmp = pd.concat(df_tmp)



df_test = pd.merge(df_test, df_tmp, on=col_index, how='left')

# df_test = pd.merge(df_test, df_test_study[[col_groupby, 'num_images']], on=col_groupby, how='left')

df_test['slice_location'] = df_test['series_index'] / (df_test['num_images'] - 1)

df_test.head()
# get weighted mean prediction per slice location

bins = 8

df_test['bins'] = bins-1

for i in range(bins):

    df_test['bins'][(df_test['slice_location']>=(i/bins)) & (df_test['slice_location']<((i+1)/bins))] = i

df_test['q_weighted_means'] = df_test['bins'].apply(lambda x: q_weighted_means[x])

df_test.head()
df_sub_tmp = copy.deepcopy(df_test[[col_index, 'q_weighted_means']])

df_sub_tmp.columns = ['id', 'label']

for i, col in enumerate(col_targets[:-1]):

    df_tmp = df_test_study[[col_groupby]]

    df_tmp.columns = ['id']

    df_tmp['label'] = mean_targets[i]

    df_tmp['id'] = df_tmp['id'] + '_{}'.format(col)

    df_sub_tmp = pd.concat([df_sub_tmp, df_tmp])

df_sub_tmp = df_sub_tmp.reset_index(drop=True)

print(df_sub_tmp.shape)

df_sub_tmp.head()
df_sub = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")

print(df_sub.shape)

df_sub = pd.merge(df_sub[['id']], df_sub_tmp, on='id', how='left')

df_sub = df_sub.fillna(0.5)

df_sub.head()
df_sub.to_csv("submission.csv", index=None)
for i in range(bins):

    print(i, np.sum(df_test['bins']==i))