# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

!git clone https://github.com/ravi02512/efficientdet-keras.git
import os

os.chdir("/kaggle/working/efficientdet-keras")
train_data_df=pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

train_data_df.head()
image_id=[f"{i}.jpg" for i in train_data_df.image_id]

xmins,ymins,xmaxs,ymaxs,area=[],[],[],[],[]

for bbox in train_data_df.bbox:

    real_bbox=eval(bbox)

    

    xmin, ymin ,w ,h=real_bbox

    

    

    

    a=int(xmin+w)

    b=int(ymin+h)

    xmaxs.append(a)

    ymaxs.append(b)



    

    c=int(xmin)

    d=int(ymin)

    xmins.append(c)

    ymins.append(d)

    

    area.append(w*h)
data=pd.DataFrame()

data["filename"]=image_id

data["width"]=train_data_df.width

data["width"]=train_data_df.height



data["class"]=["wheat"]*len(image_id)



data["xmin"]=xmins

data["ymin"]=ymins



data["xmax"]=xmaxs

data["ymax"]=ymaxs



data["iscrowd"]=[1]*len(image_id)



data["area"]=area
data.head()
filtered_df=data.drop(data[(data["area"]>200000) | (data["area"]<2000)].index)
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.to_csv("train_labels.csv",index=False)
pd.read_csv("train_labels.csv")
def int64_feature(value):

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))





def int64_list_feature(value):

  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))





def bytes_feature(value):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def bytes_list_feature(value):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))





def float_list_feature(value):

  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

from __future__ import division

from __future__ import print_function

from __future__ import absolute_import



import os

import io

import pandas as pd

import tensorflow as tf



from PIL import Image

from collections import namedtuple, OrderedDict

import hashlib





# TO-DO replace this with label map

def class_text_to_int(row_label):

    if row_label == 'wheat':

        return 1

    else:

        None





def split(df, group):

    data = namedtuple('data', ['filename', 'object'])

    gb = df.groupby(group)

    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]





def create_tf_example(group, path):

    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:

        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)

    key = hashlib.sha256(encoded_jpg).hexdigest()

    

    width, height = image.size



    filename = group.filename.encode('utf8')

    image_format = b'jpg'

    xmins = []

    xmaxs = []

    ymins = []

    ymaxs = []

    classes_text = []

    classes = []

    iscrowd=[]

    area=[]



    for index, row in group.object.iterrows():

        xmins.append(row['xmin'] / width)

        xmaxs.append(row['xmax'] / width)

        ymins.append(row['ymin'] / height)

        ymaxs.append(row['ymax'] / height)

        iscrowd.append(row["iscrowd"])

        area.append(row["area"])

        classes_text.append(row['class'].encode('utf8'))

        classes.append(class_text_to_int(row['class']))



    tf_example = tf.train.Example(features=tf.train.Features(feature={

        'image/height': int64_feature(height),

        'image/width': int64_feature(width),

        'image/filename': bytes_feature(filename),

        'image/source_id':bytes_feature(filename),

        'image/key/sha256':bytes_feature(key.encode('utf8')),

        'image/encoded':bytes_feature(encoded_jpg),

        'image/format': bytes_feature('jpg'.encode('utf8')),

        'image/object/bbox/xmin': float_list_feature(xmins),

        'image/object/bbox/xmax': float_list_feature(xmaxs),

        'image/object/bbox/ymin': float_list_feature(ymins),

        'image/object/bbox/ymax': float_list_feature(ymaxs),

        'image/object/class/text':bytes_list_feature(classes_text),

        'image/object/class/label':int64_list_feature(classes),

        'image/object/is_crowd':int64_list_feature(iscrowd),

        'image/object/area':float_list_feature(area)

    }))

    return tf_example





def main(csv_input, train_output_path,val_output_path, image_dir):

    train_writer = tf.io.TFRecordWriter(train_output_path)

    val_writer = tf.io.TFRecordWriter(val_output_path)

    path = os.path.join(image_dir)

    examples = pd.read_csv(csv_input)

    grouped = split(examples, 'filename')

    for group in grouped[500:]:

        tf_example = create_tf_example(group, path)

        train_writer.write(tf_example.SerializeToString())

        

    for group in grouped[:500]:

        tf_example = create_tf_example(group, path)

        val_writer.write(tf_example.SerializeToString())





    train_writer.close()

    val_writer.close()

    

    train_output_path = os.path.join(os.getcwd(), train_output_path)

    val_output_path = os.path.join(os.getcwd(), val_output_path)

    

    print('Successfully created the TFRecords: {}'.format(train_output_path))

    print('Successfully created the TFRecords: {}'.format(val_output_path))





if __name__ == '__main__':

    csv_input="train_labels.csv"

    train_output_path="train_data.record"

    val_output_path="val_data.record"

    image_dir="/kaggle/input/global-wheat-detection/train"

    main(csv_input, train_output_path,val_output_path, image_dir)
os.mkdir("/kaggle/working/model")
!pip install pycocotools
import os

from absl import app

from absl import flags

from absl import logging

import tensorflow as tf

from tensorflow.python.client import device_lib

import dataloader

import hparams_config

import utils

from keras import train_lib











FLAGS={'tpu':None,'gcp_project':None,'tpu_zone':None,'eval_master':'',

      'eval_name':None,'strategy':None,'num_cores':8,'use_fake_data':False,

      'use_xla':False,'model_dir':"/kaggle/working/model",'hparams':'','batch_size':2,

      'eval_samples':2,'iterations_per_loop':100,'training_file_pattern':"train_data.record",

      'validation_file_pattern':"val_data.record",'val_json_file':None,'testdev_dir':None,

      'num_examples_per_epoch':50,'num_epochs':None,'mode':'train',

      'model_name':'efficientdet-d3','eval_after_training':False,

      'debug':False,'profile':False,'min_eval_interval':180,

      'eval_timeout':None}



def get_callbacks(params,profile=False):

  tb_callback = tf.keras.callbacks.TensorBoard(

      log_dir=params['model_dir'], profile_batch=2 if profile else 0)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(

      params['model_dir']+"/best.h5", verbose=1, save_weights_only=False,save_best_only=True)

  early_stopping = tf.keras.callbacks.EarlyStopping(

      monitor='val_loss', min_delta=0, patience=10, verbose=1)

  return [tb_callback, ckpt_callback, early_stopping]





def main(FLAGS):

  # Parse and override hparams

  config = hparams_config.get_detection_config(FLAGS["model_name"])

  config.override(FLAGS["hparams"])

  if FLAGS["num_epochs"]:  # NOTE: remove this flag after updating all docs.

    config.num_epochs = FLAGS["num_epochs"]



  # Parse image size in case it is in string format.

  config.image_size = utils.parse_image_size(config.image_size)



  if FLAGS["use_xla"] and FLAGS["strategy"] != 'tpu':

    tf.config.optimizer.set_jit(True)

    for gpu in tf.config.list_physical_devices('GPU'):

      tf.config.experimental.set_memory_growth(gpu, True)



  if FLAGS["debug"]:

    tf.config.experimental_run_functions_eagerly(True)

    tf.debugging.set_log_device_placement(True)

    tf.random.set_seed(111111)

    logging.set_verbosity(logging.DEBUG)



  if FLAGS["strategy"] == 'tpu':

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(

        FLAGS["tpu"], zone=FLAGS["tpu_zone"], project=FLAGS["gcp_project"])

    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)

    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)

    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))

  elif FLAGS["strategy"] == 'gpus':

    ds_strategy = tf.distribute.MirroredStrategy()

    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))

  else:

    if tf.config.list_physical_devices("GPU"):

      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')

    else:

      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')



  # Check data path

  if FLAGS["mode"] in ('train',

                    'train_and_eval') and FLAGS["training_file_pattern"] is None:

    raise RuntimeError('You must specify --training_file_pattern for training.')

  if FLAGS["mode"] in ('eval', 'train_and_eval'):

    if FLAGS["validation_file_pattern"] is None:

      raise RuntimeError('You must specify --validation_file_pattern '

                         'for evaluation.')



  params = dict(

      config.as_dict(),

      model_name=FLAGS["model_name"],

      iterations_per_loop=FLAGS["iterations_per_loop"],

      model_dir=FLAGS["model_dir"],

      num_examples_per_epoch=FLAGS["num_examples_per_epoch"],

      strategy=FLAGS["strategy"],

      batch_size=FLAGS["batch_size"] // ds_strategy.num_replicas_in_sync,

      num_shards=ds_strategy.num_replicas_in_sync,

      val_json_file=FLAGS["val_json_file"],

      testdev_dir=FLAGS["testdev_dir"],

      mode=FLAGS["mode"])



  # set mixed precision policy by keras api.

  precision = utils.get_precision(params['strategy'], params['mixed_precision'])

  policy = tf.keras.mixed_precision.experimental.Policy(precision)

  tf.keras.mixed_precision.experimental.set_policy(policy)



  def get_dataset(is_training, params):

    file_pattern = (

        FLAGS["training_file_pattern"]

        if is_training else FLAGS["validation_file_pattern"])

    return dataloader.InputReader(

        file_pattern,

        is_training=is_training,

        use_fake_data=FLAGS["use_fake_data"],

        max_instances_per_image=config.max_instances_per_image)(

            params)



  with ds_strategy.scope():

    model = train_lib.EfficientDetNetTrain(params['model_name'], config)

    height, width = utils.parse_image_size(params['image_size'])

    model.build((params['batch_size'], height, width, 3))

    model.summary()

    model.compile(

        optimizer=train_lib.get_optimizer(params),

        loss={

            'box_loss':

                train_lib.BoxLoss(

                    params['delta'],reduction=tf.keras.losses.Reduction.NONE),

            'box_iou_loss':

                train_lib.BoxIouLoss(

                    params['iou_loss_type'],

                    params['min_level'],

                    params['max_level'],

                    params['num_scales'],

                    params['aspect_ratios'],

                    params['anchor_scale'],

                    params['image_size'],

                    reduction=tf.keras.losses.Reduction.NONE),

            'class_loss':

                train_lib.FocalLoss(

                    params['alpha'],

                    params['gamma'],

                    label_smoothing=params['label_smoothing'],

                    reduction=tf.keras.losses.Reduction.NONE)

        })

    ckpt_path = tf.train.latest_checkpoint(FLAGS["model_dir"])

    if ckpt_path:

      model.load_weights(ckpt_path)

    

    model.freeze_vars(params['var_freeze_expr'])

    model.fit(

        get_dataset(True, params=params),

        steps_per_epoch=FLAGS["num_examples_per_epoch"],

        epochs=1,

        callbacks=get_callbacks(params, FLAGS["profile"]),

        validation_data=get_dataset(False, params=params),

        validation_steps=FLAGS["eval_samples"])

    model.save_weights(os.path.join(FLAGS["model_dir"], 'model'))





if __name__ == '__main__':

  main(FLAGS)
