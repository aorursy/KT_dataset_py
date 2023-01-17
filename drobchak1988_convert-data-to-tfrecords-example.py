!ls /kaggle/input/tfrecords-gender/data/train/
import os, sys, math, time, warnings, gc, re, random

import pandas as pd

import numpy as np

from sklearn.utils import shuffle

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
def get_len(l):

    if random.randint(1,101) % 2 == 0:

        return int(l * 0.97)

    else:

        return int(l * 1.03)

def get_binry_gendor(x):

    if x == 'M':

        return 1

    else:

        return 0

def get_valid_data(PATH_DATA_CSV):

    df = pd.read_csv(PATH_DATA_CSV, iterator=True)

    result_df = pd.DataFrame(columns = ['gender','email'])

    for i in range(1): # This range(1) for example if do you want preprocessed all data you needed set range(4)

        data = df.get_chunk(50000) # And add df.get_chunk(5000000)

        data = shuffle(data)

        data = data[(data['gender'] != 'N')]

        data = data[(data['gender'] == 'M') | (data['gender'] == 'F')]

        data['email'] = data['email'].map(lambda x: valid_string(x))

        data = data.dropna()



        M = data[(data['gender'] == "M")]

        F = data[(data['gender'] == "F")]



        if len(M) > len(F):

            M = M[:get_len(len(F))]

        else:

            F = F[:get_len(len(M))]

        data = pd.concat([M, F])

        del M, F

        gc.collect()

        data['gender'] = data['gender'].map(lambda x: get_binry_gendor(x))

        data = shuffle(data)

        result_df = pd.concat([result_df, data]) 

    return result_df
chars = {"<PAD>":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,

      "g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,

      "n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,

      "u":21,"v":22,"w":23,"x":24,"y":25,"z":26,".": 27}



def valid_string(email):

    char_list = list(email.lower())

    valid_email = ''

    for c in char_list:

        if  ord(c.lower()) >= 97 and  ord(c.lower()) <= 122:

            valid_email += c.lower()

        else:

            valid_email += ' '

    valid_email = valid_email.lstrip()

    valid_email = valid_email.rstrip()

    valid_email = re.sub(" +", ".", valid_email)

    return valid_email[:15]



def conv2vec(name,no_chars):

    vec = []

    for char in valid_string(name):

        index = chars[char]

        vec.append(index)

    if len(vec) < no_chars:

        diff = no_chars - len(vec)

        vec+= [0]*(diff)

    return vec





def preprocess_to_vec(PATH_DATA_CSV):

    data_df = get_valid_data(PATH_DATA_CSV)

    data_df = shuffle(data_df) 

    data = data_df.values

    del data_df

    gc.collect()

    no_chars = 0

    for item in data:

        name = str(item[1])

        if len(name) > no_chars: no_chars = len(name)

    newData = []

    for item in data:

        name = str(item[1])

        name = conv2vec(name,no_chars)

        newData.append(name + [item[0]])

    npData = np.array(newData)

    np.random.shuffle(npData)

    x = npData[0:,0:15]

    y = npData[0:,15]

    return x, y
!mkdir tfrecords
GCS_OUTPUT= './tfrecords/'

name_size=15



def _int_feature(list_of_ints): # int64

    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def to_tfrecord(tfrec_filewriter, email, label):

    feature = {

      "email": _int_feature(email.tolist()),

      "label": _int_feature([label])

    }

    return tf.train.Example(features=tf.train.Features(feature=feature))



print("Writing TFRecords")



border=95000000 #  unique preprocessed emails ~ 95000000 

border_counter=0

batch_size = 102400

max_batches = border//batch_size

count=0

batch_caunt = 1

LIST_DATA = os.listdir('/kaggle/input/tfrecords-gender/data/train/')

while border > border_counter:

    x,y = preprocess_to_vec(f'/kaggle/input/tfrecords-gender/data/train/{LIST_DATA[batch_caunt-1]}')

    max_packing = len(x) // 102400

    count_packing = 1

    batch_last_size = len(x) % 102400

    batch = []



    for item in zip(x,y):

        count+=1

        batch.append([*item])

        if max_packing <= count_packing and len(batch) >= batch_last_size:

            filename = GCS_OUTPUT + "{:02d}-{}.tfrec".format(batch_caunt, batch_size)

            count_packing+=1

            with tf.io.TFRecordWriter(filename) as out_file:

                for i_batch in batch:

                    example = to_tfrecord(out_file,i_batch[0],i_batch[1])

                    out_file.write(example.SerializeToString())

                batch = []

                batch_caunt+=1

                count_packing = 1

            print("Wrote last file {} containing {} records".format(filename, batch_size))

        

        if len(batch) >= batch_size:

            filename = GCS_OUTPUT + "{:02d}-{}.tfrec".format(batch_caunt, batch_size)

            count_packing+=1

            with tf.io.TFRecordWriter(filename) as out_file:

                for i_batch in batch:

                    example = to_tfrecord(out_file,i_batch[0],i_batch[1])

                    out_file.write(example.SerializeToString())

                batch = []

                batch_caunt+=1

            print("Wrote file {} containing {} records".format(filename, batch_size))



    del x

    del y

    gc.collect()

    border_counter+=20000000
def read_tfrecord(example):

    features = {

        "email":tf.io.FixedLenFeature([15], tf.int64), 

        "label":tf.io.FixedLenFeature([], tf.int64, default_value=0)

        }

    example=tf.io.parse_single_example(example, features)

    return example['email'], example['label']



def load_dataset(filenames):

    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    return dataset



def get_training_dataset():

    dataset = load_dataset('./tfrecords/01-102400.tfrec')

    dataset = dataset.shuffle(25021)

    return dataset.prefetch(AUTO)



dat = get_training_dataset()

for item in dat.take(10):

    print(item[0].numpy(),item[1].numpy())