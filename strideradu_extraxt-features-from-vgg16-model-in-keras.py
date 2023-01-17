import tensorflow as tf

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.layers import Flatten, Input

import pandas as pd

from tqdm import tqdm

import numpy as np
def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

    """Wrapper for inserting float features into Example proto."""    

    if not isinstance(value, list):

        value = [value]    

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def _int64_feature(value):

    if not isinstance(value, list):

        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
train_path = "../input/train-jpg/"

test_path = "../input/test-jpg/"

train = pd.read_csv("../input/train_v2.csv")

test = pd.read_csv("../input/sample_submission_v2.csv")
# use vgg 16 model extract feature from fc1 layer

base_model = VGG16(weights='imagenet', pooling = max)

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)



flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))



label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}



tfrecords_filename = "vgg16_fc1_train.tfrecord"

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for f, tags in tqdm(train.values[:], miniters=1000):

    # preprocess input image

    img_path = train_path + "{}.jpg".format(f)

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)



    # generate feature [4096]

    features = model.predict(x)

    np.squeeze(features)



    targets = []

    for t in tags.split(' '):

        targets.append(label_map[t])



    example = tf.train.Example(features=tf.train.Features(feature={

        'video_id': _bytes_feature(f.encode('utf-8')),

        'labels': _int64_feature(targets),

        'rgb': _float_feature(features.tolist()[0])}))



    writer.write(example.SerializeToString())



writer.close()