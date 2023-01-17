!pip install tf-slim

import os

import numpy as np

from PIL import Image

from imageio import imread

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tf_slim as slim

from tf_slim.nets import inception

import tf_slim as slim

import cv2

import matplotlib.pyplot as plt
ckpt_path = "/kaggle/input/inception_v3.ckpt"

images_path = "/kaggle/input/animals/*"

img_width = 299

img_height = 299

batch_size = 16

batch_shape = [batch_size, img_height, img_width, 3]

num_classes = 1001

predict_output = []

class_names_path = "/kaggle/input/imagenet_class_names.txt"

with open(class_names_path) as f:

    class_names = f.readlines()
X = tf.placeholder(tf.float32, shape=batch_shape)



with slim.arg_scope(inception.inception_v3_arg_scope()):

    logits, end_points = inception.inception_v3(

        X, num_classes=num_classes, is_training=False

    )



predictions = end_points["Predictions"]

saver = tf.train.Saver(slim.get_model_variables())
def load_images(input_dir):

    global batch_shape

    images = np.zeros(batch_shape)

    filenames = []

    idx = 0

    batch_size = batch_shape[0]

    files = tf.gfile.Glob(input_dir)[:20]

    files.sort()

    for filepath in files:

        with tf.gfile.Open(filepath, "rb") as f:

            imgRaw = np.array(Image.fromarray(imread(f, as_gray=False, pilmode="RGB")).resize((299, 299))).astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.

        images[idx, :, :, :] = imgRaw * 2.0 - 1.0

        filenames.append(os.path.basename(filepath))

        idx += 1

        if idx == batch_size:

            yield filenames, images

            filenames = []

            images = np.zeros(batch_shape)

            idx = 0

    if idx > 0:

        yield filenames, images
session_creator = tf.train.ChiefSessionCreator(

        scaffold=tf.train.Scaffold(saver=saver),

        checkpoint_filename_with_path=ckpt_path,

        master='')
with tf.train.MonitoredSession(session_creator=session_creator) as sess:

    for filenames, images in load_images(images_path):

        labels = sess.run(predictions, feed_dict={X: images})

        for filename, label, image in zip(filenames, labels, images):

            predict_output.append([filename, label, image])
for x in predict_output:

    out_list = list(x[1])

    topPredict = sorted(range(len(out_list)), key=lambda i: out_list[i], reverse=True)[:5]

    plt.imshow((((x[2]+1)/2)*255).astype(int))

    plt.show()

    print("Filename:",x[0])

    print("Displaying the top 5 Predictions for above image:")

    for p in topPredict:

        print(class_names[p-1].strip())