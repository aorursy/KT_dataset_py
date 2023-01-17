!conda install -c conda-forge gdcm -y
#This script will be used create masks on the images prior to further processing. This will speed up the process

#of training.



import ct_scan_processing as ct

import osic_utils

import numpy as np

import tensorflow as tf

import pandas as pd

import os

import pickle

import pydicom

import gzip

from time import time as time



dir_img = 'train-image-processed'

if dir_img not in os.listdir():

    os.mkdir(dir_img)



# obtain the ids associated with the training set

train_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

patient_dcm_dict = {}

root_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

remove_ids = ["ID00078637202199415319443", "ID00105637202208831864134"]

for dirname, _, filenames in os.walk(root_dir):

    if 'ID' in dirname:

        dirname_ = dirname.replace(root_dir, "")

        # check if the id is among the ids to be removed

        s = [i for i in remove_ids if dirname_ == i]

        if len(s) == 0:            

            patient_dcm_dict[dirname_] = filenames

        

patient_dcm_dict = {k: i for i, k in enumerate(sorted(patient_dcm_dict.keys()))}



dataset = osic_utils.DatasetGen(

    patient_dcm_dict, 

    split = None, seed = 300, 

    batch_size = 1, 

    root_dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/',

    shuffle = False)



def add_img_and_stats(path):

    imgs = [ct.image_preprocess(i) for i in path]

    #return imgs



    # calculating lung volume

    lung_vol_out = []

    for masked_img, st, ps, lps in imgs:

        lung_vol_out.append(

        ct.get_volume(masked_img, tf.constant(st), tf.constant(ps), lps)

        )



    # calculating lung image statistics

    stats = []

    for img, _ in lung_vol_out:

        stats.append(

            ct.calculate_statistics(tf.constant(img))[None, ...])

    stats = tf.concat(stats, axis = 0)



    ### return lung volume, mean, variance, skew and kurtosis in that order

    # divide by the magnitude to decrease range

    lung_volume = tf.reshape(

        tf.concat([i[1][None, ...] for i in lung_vol_out], axis = 0) /1e5, 

        [-1, 1])

    all_feats = tf.cast(

        tf.concat([lung_volume, stats], axis = 1), 

        dtype = tf.float32)

    #all_feats = tf.gather(all_feats, idx, axis = 0)



    return tf.constant(imgs[0][0], dtype = 'float32'), all_feats



dataset = dataset.train.map(

    lambda path, _: tf.py_function(

        add_img_and_stats, 

        [path], 

        (tf.float32, tf.float32))

    )



foo = list(dataset.take(1))







#foo = add_img_and_stats(

#    tf.reshape(

#        tf.constant("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"), [-1,]))
start = time()

lung_stats = []



for i, tup in enumerate(dataset):

    lung_stats.append(tup[1])

    path = dir_img + '/img_proc_'+ list(patient_dcm_dict.keys())[i] + '.npy.gz'

    f = gzip.GzipFile(path, "w")

    np.save(file=f, arr=tup[0])

    f.close()

end = time()



print((end - start)/60, 'mins')
# prepare data frame for lung image statistics

df_stats = pd.DataFrame(tf.concat(lung_stats, axis = 0).numpy(), 

             columns = ['lung_vol', 'mean', 'var', 'skew', 'kurt'])

df_stats['PatientId'] = list(patient_dcm_dict.keys())[0:df_stats.shape[0]]

df_stats.head()

df_stats.to_csv("lung_statistics.csv", index = False)
#os.path.getsize('train-image-processed/')
# will revisit this image

"/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00105637202208831864134/"