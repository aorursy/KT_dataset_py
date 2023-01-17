#Below are how I convert the images and their corresponding masks into tfrecords file.

#This is inspired from Daniil's great blog on ML and CV. http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

#Note that I am not able to write the converted tfrecords file here in Kaggle so the 2nd and 3rd blocks raised error here. 

#However, you can run them in your local machine. Any hints on how to write to Kaggle's data directory will be appreciated. 
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os



#construct list for image paths and their corresponding masks in the same order

#image_path = "../data/train/*.jpg"

image_path = []

for filename in os.listdir("../input/train/"):

    image_path.append("../input/train/"+filename)



mask_path = []

for i, filename in enumerate(image_path):

    mask_path.append(filename[0:13]+"_masks/" + filename[14:-4] + "_mask.gif")
# path to where the tfrecords will be stored (change to your customized path).

# I can't run this here in Kaggle because of permission. 

# Any comment on how to write temporary files in Kaggle will be appreciated.

tfrecords_filename = '../input/train/training_data.tfrecords'

#get a writer for the tfrecord file.

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

#write data/masks into tfrecords

for i in range(len(image_path)):

    img = np.array(mpimg.imread(image_path[i]))

    mask = np.array(mpimg.imread(mask_path[i]))[:,:,0]



    height = img.shape[0]

    width = img.shape[1]

    img_raw = img.tostring()

    mask_raw = mask.tostring()

    #save the heights and widths as well so, which 

    #are needed when decoding from tfrecords back to images

    example = tf.train.Example(features=tf.train.Features(feature={

                                                          'height': _int64_feature(height),

                                                          'width': _int64_feature(width),

                                                          'image_raw': _bytes_feature(img_raw),

                                                          'mask_raw': _bytes_feature(mask_raw)}

                                                          ))

    writer.write(example.SerializeToString())

writer.close()
#run the following to verify the created tfrecord file.

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:

    example = tf.train.Example()

    example.ParseFromString(string_record)

    height = int(example.features.feature['height'].int64_list.value[0])

    width = int(example.features.feature['width'].int64_list.value[0])

    img_string = (example.features.feature['image_raw'].bytes_list.value[0])

    mask_string = (example.features.feature['mask_raw'].bytes_list.value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)

    mask_1d = np.fromstring(mask_string, dtype=np.uint8)

    #reshape back to their original shape from a 1D array read from tfrecords

    img = img_1d.reshape((height, width, -1))

    mask = mask_1d.reshape((height, width))

    plt.imshow(img)

    plt.show()

    plt.imshow(mask)

    plt.show()