import PIL

import os, re, time, tqdm

import numpy as np

import pandas as pd

from sklearn import model_selection

from collections import abc

import matplotlib.pyplot as plt

import cv2

import pydicom

import tensorflow as tf
IMG_SIZE = 512

N_SHARDS_TRAIN = 15

N_SHARDS_TEST = 1
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
def maybe_list(x):

    if isinstance(x, list):

        return x

    elif isinstance(x,  abc.Iterable):

        return list(x)

    return [x]





def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=maybe_list(value)))





def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=maybe_list(value)))





def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=maybe_list(value)))
def serialize_example(pid, img, age, sex, smoking_status, fvc):

    fvc = np.array(fvc)

    feature = {

        'patient_id':     _bytes_feature([pid]),

        'image':          _bytes_feature([img]),

        'age':            _int64_feature(age),

        'sex':            _int64_feature(sex),

        'smoking_status': _int64_feature(smoking_status),

        'FVC':            _float_feature(fvc)          

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
os.mkdir("../working/train")

os.mkdir("../working/test")
pids = os.listdir("../input/osic-pulmonary-fibrosis-progression/train/")

n = []

for pid in pids:

    folder = os.path.join("../input/osic-pulmonary-fibrosis-progression/train/", pid)

    n.append(len(os.listdir(folder)))

P = pd.DataFrame({'Patient': pids, 'n': n})

P = P.merge(df_train[['Patient', 'Age', 'FVC']].groupby('Patient').mean(), on='Patient')

grouping = [(2, 'n'), (3, 'FVC')]

for N, c in grouping:

    P['q_%s' % c] = np.array([(P[c] >= P[c].quantile(i / N)).astype('int') for i in range(1, N)]).sum(0)

P['group'] = np.array([P['q_%s' % c] * 10 ** i for i, (_, c) in enumerate(grouping)]).sum(0)

print("%d groups" % len(P.group.drop_duplicates())) 
df_train_shards = pd.DataFrame()

for shard, (_, idx) in enumerate(model_selection.StratifiedKFold(N_SHARDS_TRAIN, shuffle=True, random_state=42).split(P.Patient, P.group)):

    shard_pids = P.Patient.iloc[idx]

    print("Shard %d" % shard)

    df_train_shards = pd.concat([df_train_shards, pd.DataFrame({'shard': shard, 'Patient': shard_pids})])

    examples = []

    for pid in shard_pids:

        folder = os.path.join("../input/osic-pulmonary-fibrosis-progression/train/", pid)

        i = 1

        file = os.path.join(folder, "%d.dcm" % i)

        fvc = df_train[df_train.Patient == pid].FVC

        age = df_train[df_train.Patient == pid].Age.iloc[0]

        smoking_status = df_train[df_train.Patient == pid].SmokingStatus.map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}).iloc[0]

        sex = df_train[df_train.Patient == pid].Sex.map({'Female': 0, 'Male': 1}).iloc[0]

        try:

            while os.path.exists(file):    

                di = pydicom.dcmread(file)

                img = di.pixel_array

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape((IMG_SIZE, IMG_SIZE, 1))

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

                i += 1

                file = os.path.join(folder, "%d.dcm" % i)

                example = serialize_example(str.encode(pid), img, age, sex, smoking_status, fvc)

                examples.append(example)

        except RuntimeError:

            pass

        pass

    with tf.io.TFRecordWriter(os.path.join("../working/train", "train%.2d-%d.tfrec" % (shard, len(examples)))) as writer:

        for example in examples:

            writer.write(example)

            pass

        pass

    pass
xxx = df_train_shards.merge(df_train, on='Patient')

c = 'FVC'

plt.figure(figsize=(15, 10))

for shard in xxx.shard.drop_duplicates():

    plt.subplot(3, 5, shard + 1)

    plt.hist(xxx[xxx.shard == shard][c], bins=10, density=True, range=(xxx[c].min(), xxx[c].max()), alpha=0.5, histtype='step')
pids = os.listdir("../input/osic-pulmonary-fibrosis-progression/test/")

n = []

for pid in pids:

    folder = os.path.join("../input/osic-pulmonary-fibrosis-progression/test/", pid)

    n.append(len(os.listdir(folder)))

P = pd.DataFrame({'Patient': pids, 'n': n})

P = P.merge(df_train[['Patient', 'Age', 'FVC']].groupby('Patient').mean(), on='Patient')

grouping = [(2, 'n'), (3, 'FVC')]

for N, c in grouping:

    P['q_%s' % c] = np.array([(P[c] >= P[c].quantile(i / N)).astype('int') for i in range(1, N)]).sum(0)

P['group'] = np.array([P['q_%s' % c] * 10 ** i for i, (_, c) in enumerate(grouping)]).sum(0)

print("%d groups" % len(P.group.drop_duplicates())) 
df_test_shards = pd.DataFrame()

# for shard, (_, idx) in enumerate(model_selection.StratifiedKFold(N_SHARDS_TEST, shuffle=True, random_state=42).split(P.Patient, P.group)):

for shard, (_, idx) in [(0, (None, list(range(P.shape[0]))))]:

    shard_pids = P.Patient.iloc[idx]

    print("Shard %d" % shard)

    df_test_shards = pd.concat([df_test_shards, pd.DataFrame({'shard': shard, 'Patient': shard_pids})])

    examples = []

    for pid in shard_pids:

        folder = os.path.join("../input/osic-pulmonary-fibrosis-progression/test/", pid)

        i = 1

        file = os.path.join(folder, "%d.dcm" % i)

        fvc = df_test[df_test.Patient == pid].FVC

        age = df_test[df_test.Patient == pid].Age.iloc[0]

        smoking_status = df_test[df_test.Patient == pid].SmokingStatus.map({'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}).iloc[0]

        sex = df_test[df_test.Patient == pid].Sex.map({'Female': 0, 'Male': 1}).iloc[0]

        try:

            while os.path.exists(file):    

                di = pydicom.dcmread(file)

                img = di.pixel_array

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape((IMG_SIZE, IMG_SIZE, 1))

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

                i += 1

                file = os.path.join(folder, "%d.dcm" % i)

                example = serialize_example(str.encode(pid), img, age, sex, smoking_status, fvc)

                examples.append(example)

        except RuntimeError:

            pass

        pass

    with tf.io.TFRecordWriter(os.path.join("../working/test", "test%.2d-%d.tfrec" % (shard, len(examples)))) as writer:

        for example in examples:

            writer.write(example)

            pass

        pass

    pass
df_train_shards.to_csv("train_shards.csv", index=False)

df_test_shards.to_csv("test_shards.csv", index=False)