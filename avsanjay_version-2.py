# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import csv

import os



import numpy as np

import pandas as pd

from scipy.misc import imread

from scipy.misc import imsave



import tensorflow as tf

from tensorflow.contrib.slim.nets import inception



slim = tf.contrib.slim



import sys



#sys.path.insert(0, r"./cleverhans")



import sys

sys.path.insert(0, r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master")



from cleverhans.attacks import FastGradientMethod

from cleverhans.attacks import BasicIterativeMethod

from cleverhans.attacks import CarliniWagnerL2



tensorflow_master = ""



input_dir = r"./input"





output_dir = "./output"

tf.flags.DEFINE_string(

    'master', '', 'The address of the TensorFlow master to use.')



tf.flags.DEFINE_string(

    'checkpoint_path', '/home/nrajeshrao/projects/kaggle/nips_2017/kagglesubmission1/inception_v3.ckpt',

    'Path to checkpoint for inception network.')



tf.flags.DEFINE_string(

    'input_dir', '/home/nrajeshrao/projects/kaggle/nips_2017/kagglesubmission1/input', 'Input directory with images.')



tf.flags.DEFINE_string(

    'output_dir', '/home/nrajeshrao/projects/kaggle/nips_2017/kagglesubmission1/output',

    'Output directory with images.')



tf.flags.DEFINE_float(

    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')



tf.flags.DEFINE_integer(

    'image_width', 299, 'Width of each input images.')



tf.flags.DEFINE_integer(

    'image_height', 299, 'Height of each input images.')



tf.flags.DEFINE_integer(

    'batch_size', 16, 'How many images process at one time.')



FLAGS = tf.flags.FLAGS





def load_target_class(input_dir):

    """Loads target classes."""

    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:

        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}





def load_images(input_dir, batch_shape):

    """Read png images from input directory in batches.

  

    Args:

      input_dir: input directory

      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  

    Yields:

      filenames: list file names without path of each image

        Lenght of this list could be less than batch_size, in this case only

        first few images of the result are elements of the minibatch.

      images: array with all images from this batch

    """

    # input_dir = r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_attacks\fgsm\input"

    images = np.zeros(batch_shape)

    filenames = []

    idx = 0

    batch_size = batch_shape[0]

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):

        # with tf.gfile.Open(filepath) as f:

        # image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.

        # images[idx, :, :, :] = image * 2.0 - 1.0

        with tf.gfile.Open(filepath, "rb") as f:

            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) * 2.0 / 255.0 - 1.0



        filenames.append(os.path.basename(filepath))

        idx += 1

        if idx == batch_size:

            yield filenames, images

            filenames = []

            images = np.zeros(batch_shape)

            idx = 0

    if idx > 0:

        yield filenames, images





def save_images(images, filenames, output_dir):

    """Saves images to the output directory.

  

    Args:

      images: array with minibatch of images

      filenames: list of filenames without path

        If number of file names in this list less than number of images in

        the minibatch then only first len(filenames) images will be saved.

      output_dir: directory where to save images

    """

    for i, filename in enumerate(filenames):

        # Images for inception classifier are normalized to be in [-1, 1] interval,

        # so rescale them back to [0, 1].

        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:

            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')





globLogits = 0

globEnd_points = 0





class InceptionModel(object):

    """Model class for CleverHans library."""



    def __init__(self, num_classes):

        self.num_classes = num_classes

        self.built = False



    def __call__(self, x_input):

        """Constructs model and return probabilities for given input."""

        reuse = True if self.built else None

        with slim.arg_scope(inception.inception_v3_arg_scope()):

            logits, end_points = inception.inception_v3(

                x_input, num_classes=self.num_classes, is_training=False,

                reuse=reuse)

        global globLogits

        global globEnd_points



        # globLogits = logits

        # globEnd_points = end_points

        self.built = True

        output = end_points['Predictions']

        # Strip off the extra reshape op at the output

        probs = output.op.inputs[0]

        return probs





eps = 2.0 * FLAGS.max_epsilon / 255.0

batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

num_classes = 1001



categories = pd.read_csv(r"./input/categories.csv")



image_classes = pd.read_csv(r"./input/images.csv")



all_images_target_class = {image_classes["ImageId"][i] + ".png": image_classes["TargetClass"][i] for i in

                           image_classes.index}



output = pd.DataFrame.from_dict(all_images_target_class, orient='index')

output.to_csv(

    r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\target_class.csv")



image_metadata = pd.DataFrame()



for filenames, images in load_images(input_dir, batch_shape):

    image_metadata1 = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")

    frames = [image_metadata, image_metadata1]

    image_metadata = pd.concat(frames)



image_metadata2 = image_metadata

image_metadata2['PredictedClass'] = 0



image_metadata2.index = range(len(image_metadata2.index))



with tf.Graph().as_default():

    # Prepare graph

    x_input = tf.placeholder(tf.float32, shape=batch_shape)



    model = InceptionModel(num_classes)



    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

    one_hot_target_class = tf.one_hot(target_class_input, num_classes)



    # session_creator = tf.train.ChiefSessionCreator(

    #    scaffold=tf.train.Scaffold(saver=saver),

    #    checkpoint_filename_with_path=FLAGS.checkpoint_path,

    #    master=FLAGS.master)





    # https://stackoverflow.com/questions/43245231/how-do-monitored-training-sessions-work

    # with tf.train.MonitoredSession(session_creator=session_creator) as sess:

    with tf.Session() as sess:

        basicIterative = BasicIterativeMethod(model, sess=sess)

        x_adv1 = basicIterative.generate(x_input, y_target=one_hot_target_class, eps=eps, clip_min=-1., clip_max=1.)



        carliniWagner = CarliniWagnerL2(model, sess=sess)

        x_adv2 = carliniWagner.generate(x_input, y_target=one_hot_target_class, clip_min=-1., clip_max=1.)



        x_adv = (x_adv1 + x_adv2) / 2

        x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)



        saver = tf.train.Saver(slim.get_model_variables())



        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)

        sess.run(tf.global_variables_initializer())

        for filenames, images in load_images(input_dir, batch_shape):

            print("looping thru.... i should have multiple of this printouts")

            target_class_for_batch = (

                [all_images_target_class[n] for n in filenames]

                + [0] * (FLAGS.batch_size - len(filenames)))



            targeted_images = sess.run(x_adv,

                                       feed_dict={

                                           x_input: images,

                                           target_class_input: target_class_for_batch})



            save_images(targeted_images, filenames, output_dir)