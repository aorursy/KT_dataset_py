import os



os.listdir('/kaggle/input/some-pictures-that-i-have-taken')
from PIL import Image



if 'mytfrs' not in os.listdir():

    os.mkdir('mytfrs')



path = '/kaggle/input/some-pictures-that-i-have-taken'

for f in os.listdir(path):

    fp = os.path.join(path,f)

    im = Image.open(fp)

    im = im.resize((256,256))

    im.save(os.path.join('mytfrs',f))
os.listdir(path)
import os

import matplotlib.image as mpimg

import tensorflow as tf



class GenerateTFRecord:

    def __init__(self, labels):

        self.labels = labels



    def convert_image_folder(self, img_folder, tfrecord_file_name):

        # Get all file names of images present in folder

        img_paths = os.listdir(img_folder)

        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_paths]



        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:

            for img_path in img_paths:

                example = self._convert_image(img_path)

                writer.write(example.SerializeToString())



    def _convert_image(self, img_path):

        #label = self._get_label_with_filename(img_path)

        img_shape = mpimg.imread(img_path).shape

        filename = os.path.basename(img_path)



        # Read image data in terms of bytes

        with tf.compat.v1.gfile.FastGFile(img_path, 'rb') as fid:

            image_data = fid.read()

        

        example = tf.train.Example(features = tf.train.Features(feature = {

            'image_name': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),

            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),

            'target': tf.train.Feature(bytes_list = tf.train.BytesList(value = ['meh'.encode('utf-8')]))

        }))

        return example



    def _get_label_with_filename(self, filename):

        basename = os.path.basename(filename).split('.')[0]

        basename = basename.split('_')[0]

        return self.labels[basename]





t = GenerateTFRecord({0:'yo'})

t.convert_image_folder('mytfrs', 'my.tfrecord')
os.listdir()