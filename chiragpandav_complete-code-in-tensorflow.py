# everything is here

# https://github.com/chiragp007/Object-detction-With-TensorFlow
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import cv2
import os
import re

path_train_csv="/home/chirag/Downloads/Kaggle_project/train.csv"
img_dir_all_data="/home/chirag/Downloads/Kaggle_project/My_project/train"
train_df=pd.read_csv(path_train_csv)


def split_Bbox(value):
    reg = np.array(re.findall("([0-9]+[.]?[0-9]*)", value))
    if len(reg) == 0:
        reg = [-1, -1, -1, -1]
        print(reg)
    return reg

def csv_manipulation_and_filtering(df):
    df['xmin'] = 0
    df['ymin'] = 0
    df['xmax'] = 0
    df['ymax'] = 0
    df["width"]=600
    df["height"]=600
    
    df=df.rename(columns={"image_id": "filename"})
    df=df.rename(columns={"source": "class"})
    df[['xmin', 'ymin', 'xmax', 'ymax']] = np.stack(df['bbox'].apply(lambda value: split_Bbox(value)))
    df.drop(columns=['bbox'], inplace=True)
        
    df['class']= "wheat_head_detected"
    df["filename"]=df['filename'].astype(str)+'.jpg'
    
    df['xmin'] = df['xmin'].astype(np.float)
    df['ymin'] = df['ymin'].astype(np.float)
    df['xmax'] = df['xmax'].astype(np.float)
    df['ymax'] = df['ymax'].astype(np.float)
    df['xmax']=df['xmin']+df['xmax']
    df['ymax']=df['ymin']+df['ymax']
  
        
    return df


df=csv_manipulation_and_filtering(train_df)
df.head()
## code for finding images which do not have BBOXs ## 

uniq=df["filename"].unique()
from PIL import Image
import glob
fileList = []
count=0
for root, dirs, files in os.walk(img_dir_all_data, topdown=False):
    for name in files:
        if name.endswith('.jpg'):
            name_of_img=os.path.splitext(name)[0]
            count+=1
            if name_of_img not in uniq:
                fileList.append(name_of_img) 
                
                
len(fileList)   

# REMOVE THESE IMAGES FROM DATASET !!
import sys
sys.path.append("/home/chirag/Downloads/models-master/research/slim/")
from object_detection.legacy import train
df.head()
test_csv=df[0:47]
train_csv=df[47:]
test_csv.tail()
train_csv.head()
train_output_path="/home/chirag/Downloads/Kaggle_project/My_project/train.record"
test_output_path="/home/chirag/Downloads/Kaggle_project/My_project/test.record"

test_img_dir="/home/chirag/Downloads/Kaggle_project/images/test/"

filtered_img_path="/home/chirag/Downloads/Kaggle_project/images/train/"
#chnage your path , pass train and test dataframe
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def class_text_to_int(row_label):
    if row_label == 'wheat_head_detected':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(train_output_path)
    path = os.path.join(filtered_img_path)
    
    examples=train_csv
    
    grouped = split(examples, 'filename')
    
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(train_output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()





import functools
import json
import os
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags


FLAGS = flags.FLAGS

# Create TRAINING folder in dir
# copy this (faster_rcnn_resnet50_pets.config) and paste in this folder
# do changes accordingly 
# create  lable.pbtxt file in this folder and put path in "faster_rcnn_resnet50_pets"
# run this Cell and it will create many models in folders, it takes alot of time

traning_dir="/home/Downloads/models-master/research/object_detection/tranining"
Pipeline_config_path ="/home/Downloads/models-master/research/object_detection/tranining/faster_rcnn_resnet50_pets.config"

@contrib_framework.deprecated(None, 'Use object_detection/model_main.py.')

def main(_):
  if FLAGS.task == 0: tf.gfile.MakeDirs(traning_dir)
  if Pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        Pipeline_config_path)
    if FLAGS.task == 0:
      tf.gfile.Copy(Pipeline_config_path,
                    os.path.join(traning_dir, 'pipeline.config'),
                    overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        train_config_path=FLAGS.train_config_path,
        train_input_config_path=FLAGS.input_config_path)
    if FLAGS.task == 0:
      for name, config in [('model.config', FLAGS.model_config_path),
                           ('train.config', FLAGS.train_config_path),
                           ('input.config', FLAGS.input_config_path)]:
        tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                      overwrite=True)

  model_config = configs['model']
  train_config = configs['train_config']
  input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)

  def get_next(config):
    return dataset_builder.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=True)

  trainer.train(
      create_input_dict_fn,
      model_fn,
      train_config,
      master,
      task,
      FLAGS.num_clones,
      worker_replicas,
      FLAGS.clone_on_cpu,
      ps_tasks,
      worker_job_name,
      is_chief,
      traning_dir,
      graph_hook_fn=graph_rewriter_fn)


if __name__ == '__main__':
  tf.app.run()

# MAKE SURE, YOU ARE IN "/home/models-master/research/object_detection"
# in export_inference_graph.py, don't forget to put 
#                                "sys.path.append("/home/chirag/Downloads/models-master/research/slim/")"  
# "export_inference_graph.py " download this from my repo
#  python export_inference_graph.py 
#     --input_type image_tensor 
#     --pipeline_config_path tranining/faster_rcnn_resnet50_pets.config  
#     --trained_checkpoint_prefix tranining/model.ckpt-4310 
#     --output_directory inference_graph

## NOW YOU WILL HAVE .PB file in "inference_graph" folder

