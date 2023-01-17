%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import sys
import io
import os
import csv
import tensorflow as tf
from time import time 
from urllib import request, error
from PIL import Image
from io import BytesIO
from IPython import display
import datetime

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def write_example(writer, pil_image, key, url):
    example = tf.train.Example(features=tf.train.Features(feature={
            #'height': _int64_feature(pil_image.size[1]),
            #'width': _int64_feature(pil_image.size[0]),
            'key': _bytes_feature(str.encode(key)),
            'url': _bytes_feature(str.encode(url)),
            'img_raw':_bytes_feature(pil_image)
        }))
    writer.write(example.SerializeToString())
input_folder = "../input"
output_folder = "../working"
data_file = os.path.join(input_folder, "index.csv")
key_url_list = parse_data(data_file)
opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

tfrecords_filename = os.path.join(output_folder, 'img_{}.tfrecords')
start_time = time()

skipped = open(os.path.join(output_folder,'skipped.csv'),'w')
downloaded = open(os.path.join(output_folder, 'downloaded.csv'),'w')
counters = {}
counters['skipped'] = 0
counters['attempts'] = 0
writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(counters['attempts']), options=opts)
total = len(key_url_list)
print("starting download")
for x in key_url_list:
    key = x[0]
    url = x[1]
    is_skipped = False
    image_data = None
    if counters['attempts'] % 1000 == 0:
        if writer:
            writer.close()      
        writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(counters['attempts']), options=opts)
        downloaded.flush()
        skipped.flush()
    
    try:
        counters['attempts'] += 1
        response = request.urlopen(url)
        image_data = response.read()
        display.clear_output(wait=True)
        print("{:5}% | attempts: {:7} | skipped: {:7} | elapsed: {:8} | {:4} it/s".format(
                  round(counters['attempts']/total,2)
                , counters['attempts']
                , counters['skipped']
                , str(datetime.timedelta(seconds=round(time()-start_time,2))).split('.', 2)[0]
                , round(counters['attempts']/(time()-start_time),2)
            ))
    except:
        skipped.write('"{}","{}"\n'.format(key, url))
        counters['skipped'] += 1
        is_skipped = True
    
    if not is_skipped:
        downloaded.write('"{}","{}"\n'.format(key, url))
        write_example(writer, image_data, key, url)

    if counters['attempts']>1:
        break

if downloaded:
    downloaded.close()
if skipped:
    skipped.close()
if writer:
    writer.close()   

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename.format(0), options=opts)

img_string = ""
i=0
for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
                
    img_string = (example.features.feature['img_raw'].bytes_list.value[0])
    
    key = (example.features.feature['key']
                                .bytes_list
                                .value[0])
    
    url = (example.features.feature['url']
                                .bytes_list
                                .value[0])
    
    img = tf.image.decode_jpeg(example.features.feature['img_raw'].bytes_list.value[0], channels=3)
    img.set_shape([180,180,3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    print(key, url)
    #print(height, width)
    i = i + 1
    if i > 2:
        break   
with tf.Session() as sess:
    img_s = sess.run(img)
plt.imshow(img_s)