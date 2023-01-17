import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np

#驗證集數量
_NUM_TEST = 500

#隨機種子
_RANDOM_SEED = 0

#數據集路徑 
DATA_DTR = "C:/Users/USER/Anaconda3/envs/tensorflow/Lib/site-packages/tensorflow/captcha/images"

#tfrecord文件存放路徑
TFRECORD_DIR = "C:/Users/USER/Anaconda3/envs/tensorflow/Lib/site-packages/tensorflow/captcha"

#判斷tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
        return True

#獲取所有驗證碼圖片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        #獲取文件路徑
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol message.
    return tf.train.Example(features=tf.train.Features(feature = {
        'image':bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label2),
        'label2': int64_feature(label3),
        'label3': int64_feature(label3),
    }))

#把數據集轉為TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']
    
    with tf.Session() as sess:
        #定義tfrecord文件路徑和名字
        output_filename = os.path.join(TFRECORD_DIR, split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d%d' % (i+1, len(filenames)))
                    sys.stdout.flush()
                    
                    #讀取圖片
                    image_data = Image.open(filename)
                    #根據模型的結構resize
                    image_data = image_data_resize((224, 224))
                    #灰度化
                    image_data = np.array(image_data.convert('L'))
                    #把圖片轉成bytes
                    image_data = image_data_tobytes()
                    
                    #獲取label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                        
                    #產生protocol數據類型
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                    
                except IOError as e:
                    print('無法讀取:', filename)
                    print('錯誤:', e)