import numpy as np

import pandas as pd

from pathlib import Path

import h5py

from tqdm import tqdm



import tensorflow as tf



import matplotlib.pyplot as plt

%matplotlib inline
BASE_PATH = Path("../input/trends-assessment-prediction")

fmri_scan_folder = BASE_PATH / "fMRI_train"

train_scores_path = BASE_PATH / "train_scores.csv"
mat_files = list(fmri_scan_folder.glob("*.mat"))
df = pd.read_csv(train_scores_path)
mat_file_ids = list(map(int, [x.stem for x in mat_files]))

tmp_df = pd.DataFrame({"Id":mat_file_ids, "path":list(map(str, mat_files))})
df = df.merge(tmp_df, on="Id", how="outer")
df.isnull().sum()
# All missing target values are filled with 0.0, so this can be handled in training in the loss function

df = df.fillna(value=0.0)
brain = h5py.File(mat_files[0], "r")["SM_feature"][:]
# We do not need to store this is the TFRecords file as it is the same for every fmri scan

num_of_brains, brain_width, brain_height, brain_depth = brain.shape
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def serialize_example(_id, brain, age, domain1_var1, domain1_var2, domain2_var1, domain2_var2):

    feature_dict = {

        'id':_bytes_feature(_id),

        'brain': _bytes_feature(brain),

        'age': _float_feature(age),

        'domain1_var1': _float_feature(domain1_var1),

        'domain1_var2': _float_feature(domain1_var2),

        'domain2_var1': _float_feature(domain2_var1),

        'domain2_var2': _float_feature(domain2_var2),

    }



    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example_proto
record_file = "fmri_dataset_train.tfrecods"





with tf.io.TFRecordWriter(record_file) as writer:

    for row_index in tqdm(range(len(df))):

        row = df.iloc[row_index]



        _id = str(row["Id"]).encode()

        file_path = row["path"]

        age = row["age"]

        domain1_var1 = row["domain1_var1"]

        domain1_var2 = row["domain1_var2"]

        domain2_var1 = row["domain2_var1"]

        domain2_var2 = row["domain2_var2"]

        

        brains = h5py.File(file_path, "r")["SM_feature"][:]



        # Iterate over every brain scan and assign the same values as it is the same patient

        # Only 1 random brain scan is used

        rnd_index = np.random.randint(0, brains.shape[0])

        for brain_scan_index in range(rnd_index, rnd_index+1):

            brain = brains[brain_scan_index, :, :, :]

            brain = brain.astype(np.float16)

            brain = brain.tostring()

            

            tf_example = serialize_example(_id, brain, age, domain1_var1, domain1_var2, domain2_var1, domain2_var2)

            writer.write(tf_example.SerializeToString())
dataset = tf.data.TFRecordDataset(record_file)
dataset
feature_description = {

    'id': tf.io.FixedLenFeature([], tf.string),

    'brain': tf.io.FixedLenFeature([], tf.string),

    'age': tf.io.FixedLenFeature([], tf.float32),

    'domain1_var1': tf.io.FixedLenFeature([], tf.float32),

    'domain1_var2': tf.io.FixedLenFeature([], tf.float32),

    'domain2_var1': tf.io.FixedLenFeature([], tf.float32),

    'domain2_var2': tf.io.FixedLenFeature([], tf.float32),

    }





def parse_example(example_proto):

    return tf.io.parse_single_example(example_proto, feature_description)
dataset = dataset.map(parse_example)
for x in dataset.take(2):

    brain_flat = np.frombuffer(x["brain"].numpy(), dtype=np.float16)

    brain = brain_flat.reshape((brain_width, brain_height, brain_depth))

    

    print(f"Id: {x['id'].numpy().decode() }")

    print(f"Brain scan shape {brain.shape}")

    print(f"Age: {x['age'].numpy():.4f}")

    print(f"Domain 1 Var 1: {x['domain1_var1'].numpy():.4f}")

    print(f"Domain 1 Var 2: {x['domain1_var2'].numpy():.4f}")

    print(f"Domain 2 Var 1: {x['domain2_var1'].numpy():.4f}")

    print(f"Domain 2 Var 2: {x['domain2_var2'].numpy():.4f}")

    print("-"*30)
fig, axs = plt.subplots(5, 5, figsize=(10, 10))



for i, ax in enumerate(axs.flatten()):

    ax.imshow(brain[i, :, :].astype(np.float32))

    ax.axis("off")



plt.tight_layout()