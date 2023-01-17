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

sys.path.insert(0, r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master")



from cleverhans.attacks import FastGradientMethod

from cleverhans.attacks import BasicIterativeMethod



tensorflow_master = ""

checkpoint_path   = r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\inception_v3.ckpt"

input_dir         = r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\images"

output_dir        = r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\output"



tf.flags.DEFINE_string(

    'master', '', 'The address of the TensorFlow master to use.')



tf.flags.DEFINE_string(

    'checkpoint_path', '', 'Path to checkpoint for inception network.')



tf.flags.DEFINE_string(

    'input_dir', '', 'Input directory with images.')



tf.flags.DEFINE_string(

    'output_dir', '', 'Output directory with images.')



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

  #input_dir = r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_attacks\fgsm\input"

  images = np.zeros(batch_shape)

  filenames = []

  idx = 0

  batch_size = batch_shape[0]

  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):

    #with tf.gfile.Open(filepath) as f:

      #image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0

    # Images for inception classifier are normalized to be in [-1, 1] interval.

    #images[idx, :, :, :] = image * 2.0 - 1.0

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



    #globLogits = logits

    #globEnd_points = end_points

    self.built = True

    output = end_points['Predictions']

    # Strip off the extra reshape op at the output

    probs = output.op.inputs[0]

    return  probs









eps = 2.0 * FLAGS.max_epsilon / 255.0

batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

num_classes = 1001



categories = pd.read_csv(r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\categories.csv")

image_classes = pd.read_csv(r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\images.csv")







all_images_target_class = {image_classes["ImageId"][i] + ".png": image_classes["TargetClass"][i]

                           for i in image_classes.index}



output = pd.DataFrame.from_dict(all_images_target_class, orient = 'index')

output.to_csv(r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\target_class.csv")

#pd.DataFrame(all_images_target_class).T.reset_index().to_csv('target_class.csv',header = False, index = True)

# output = pd.DataFrame({'ID': test['ID'].astype(np.int32), 'y': y_pred})

# output.to_csv('best_pred.csv')





image_metadata = pd.DataFrame()



for filenames, images in load_images(input_dir, batch_shape):

    image_metadata1 = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes,

                                                                                   on="ImageId")

    frames = [image_metadata, image_metadata1]

    image_metadata = pd.concat(frames)



    #image_metadata = image_metadata1.append(image_metadata)







true_classes = image_metadata["TrueLabel"].tolist()

target_classes = true_labels = image_metadata["TargetClass"].tolist()

true_classes_names = (pd.DataFrame({"CategoryId": true_classes})

                        .merge(categories, on="CategoryId")["CategoryName"].tolist())

target_classes_names = (pd.DataFrame({"CategoryId": target_classes})

                          .merge(categories, on="CategoryId")["CategoryName"].tolist())



#all_images_target_class = {image_metadata["ImageId"][i] + ".png": image_metadata["TargetClass"][i]

                           #for i in image_metadata.index}

all_images_target_class = {image_classes["ImageId"][i] + ".png": image_classes["TargetClass"][i]

                           for i in image_classes.index}



#all_images_target_class.to_csv('target_class.csv')



image_metadata2 = image_metadata

image_metadata2['PredictedClass'] = 0



#image_metadata2 = image_metadata2.reset_index(drop=True)

#image_metadata2 = image_metadata2.reset_index(drop=True)



image_metadata2.index = range(len(image_metadata2.index))



#duplicated3 = image_metadata.duplicated('index')

#duplicated = image_metadata[image_metadata.duplicated(['index'],keep = False)]

#duplicated2 = image_metadata2[image_metadat2.index.duplicated()]



with tf.Graph().as_default():

    # Prepare graph

    x_input = tf.placeholder(tf.float32, shape=batch_shape)



    #model = InceptionModel(num_classes)



    target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

    one_hot_target_class = tf.one_hot(target_class_input, num_classes)



    #fgsm = FastGradientMethod(model)

    #x_adv = fgsm.generate(x_input, y_target=one_hot_target_class, eps=eps, clip_min=-1., clip_max=1.)



    #basicIterative = BasicIterativeMethod(model)

    #x_adv = basicIterative.generate(x_input, y_target=one_hot_target_class, eps=eps, clip_min=-1., clip_max=1.)



    with slim.arg_scope(inception.inception_v3_arg_scope()):

        logits, end_points = inception.inception_v3(

            x_input, num_classes=num_classes, is_training=False)



    predicted_labels = tf.argmax(end_points['Predictions'], 1)



    #logits =globLogits

    #end_points = globEnd_points



    #model = returnValue['y0']

    #logits = returnValue['y1']

    #end_points = returnValue['y2']







    #fgsm = FastGradientMethod(model)

    #x_adv1 = fgsm.generate(x_input,y_target= one_hot_target_class, eps=eps, clip_min=-1., clip_max=1.)







    #basicIterative = BasicIterativeMethod(model)

    #x_adv2 = basicIterative.generate(x_input,y_target= one_hot_target_class, eps=eps, clip_min=-1., clip_max=1.)







    #output = end_points['Predictions']

    # Strip off the extra reshape op at the output

    #probs = output.op.inputs[0]



    #model = probs



    #output = end_points['Predictions']

    #Strip off the extra reshape op at the output

    #probs = output.op.inputs[0]

    #fgsm = FastGradientMethod(probs)

    #x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)





    #target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

    #one_hot_target_class = tf.one_hot(target_class_input, num_classes)

    # cross_entropy will be a vector of size 16. One value for one input image. There are 16 input images in a batch

    # https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,

                                                    logits,

                                                    label_smoothing=0.1,

                                                    weights=1.0)

    # one_hot_target_class is of the type [16,1001]. In each row only one column belonging to the image has 1. The rest has zero

    # logits is the output of the classifier. It is also of the type [16,1001]

    # Each row contains the probability that this image belongs to one of the 1001 class

    # The contents of each row, as they are probabilities adds up to 1.

    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,

                                                     end_points['AuxLogits'],

                                                     label_smoothing=0.1,

                                                     weights=0.4)



    #cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,

    #                                                 end_points['Predictions'],

    #                                                 label_smoothing=0.1,

    #                                                 weights=0.4)



    #end_points['auxlogits] is another output similiar to logits. It is similiar to logits

    # other intermediery outputs are also available. May be we can use that.

    # https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable

    x_adv = x_input - eps * tf.sign(tf.gradients(cross_entropy, x_input)[0])

    #y = x_adv5.assert_equal_None

    #x_adv6 = x_adv2 - eps * tf.sign(tf.gradients(cross_entropy, x_adv5)[0])

    #x_adv = x_adv1 - eps * tf.sign(tf.gradients(cross_entropy, x_adv6)[0])









    # x_input is of the shape [16,299,299,3]

    # the 16 belongs to the 16 images in the batch

    # Each image has 299 rows. Each row has 299 columns. Each column has a value like [4,4,6]

    # each value is [R,G,B] vector

    # cross_entropy will be of the type [16] i.e one value for the 16 images

    # tf.gradients will take  the x_input value which is of type 299,299, 3

    # and divide each of the 3 values contained in row, column by the cross entropy value

    # if row 1 is like this [ [16,16,16], [ 8, 4, 2] , [ 2, 4, 6].......

    # if the corss entropy value for row 1 is 2

    # tf.gradient will return some thing like this

    # [ [ 8,8,8], [4,2,1] [ 1, 2, 3]......



    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)



    # clip_by_value  clips these triplet value between  -1 to 1.



    saver = tf.train.Saver(slim.get_model_variables())

    session_creator = tf.train.ChiefSessionCreator(

        scaffold=tf.train.Scaffold(saver=saver),

        checkpoint_filename_with_path=FLAGS.checkpoint_path,

        master=FLAGS.master)



    with tf.train.MonitoredSession(session_creator=session_creator) as sess:

      for filenames, images in load_images(input_dir, batch_shape):

        print("looping thru.... i should have multiple of this printouts")

        target_class_for_batch = (

            [all_images_target_class[n] for n in filenames]

            + [0] * (FLAGS.batch_size - len(filenames)))



        #target_class_for_batch2 = [all_images_target_class[n] for n in filenames]

        #target_class_for_batch3 = [all_images_target_class[n] for n in filenames] + ([1] * ((FLAGS.batch_size - len(filenames) +1)))

        #experiment = [1] * ((FLAGS.batch_size - len(filenames) +1))

        #subValue = FLAGS.batch_size - len(filenames)

        #for n in filenames:

            #target_class_temp = all_images_target_class[n]





        # both x_input and target_class_input are place holders

        # both are nodes in the graphs. Final or intermediatory

        targeted_images = sess.run(x_adv,

                              feed_dict={

                                  x_input: images,

                                  target_class_input: target_class_for_batch})



        #predicted_targeted_classes = sess.run(predicted_labels, feed_dict={x_input: targeted_images})

        #debugpoint = end_points['Predictions']

        #predicted_labels = tf.argmax(end_points['Predictions'])





        save_images(targeted_images, filenames, output_dir)



        predicted_targeted_classes = sess.run(predicted_labels, feed_dict={x_input: targeted_images})



        predicted_targeted_classes_names = (pd.DataFrame({"CategoryId": predicted_targeted_classes})

                                            .merge(categories, on="CategoryId")["CategoryName"].tolist())

        i = 0

        j = 0



  #      for j in range(len(predicted_targeted_classes)):

  #          temp = predicted_targeted_classes[j]



        #j = 0



        #image_metadata2 = image_metadata

        #image_metadata2['PredictedClass'] = 0





        print(" just before the for loop..... again i should have multiple of these print outs")

        for n in filenames:

            newfilename = n.split(".")[0]

            print(newfilename)

            for index, row in image_metadata.iterrows():

                if row['ImageId'] == newfilename:

                   #roww['PredictedClass'] = predicted_targeted_classes[j]

                   print("i am in the inner if loop")

                   print("index value is")

                   print(index)

                   image_metadata2.set_value(index,'PredictedClass',predicted_targeted_classes[j] )

                   break





  #      for n in filenames:

  #          newfilename = n.split(".")[0]

  #          for i in range(len(image_metadata2)):

  #              if image_metadata2.iloc[i]['ImageId'] == newfilename:

  #                  image_metadata2.set_value(i,'PredictedClass',predicted_targeted_classes[j] )

                    #image_metadata2.at[i,'PredictedClass'] = predicted_targeted_classes[j]

                    #image_metadata2[i]['PredictedClass'] = predicted_targeted_classes[j]

  #                  j = j+ 1

  #                  break



    k = 0

    image_metadata2.to_csv(

            r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\final_results.csv")



    all_images_target_class2 = {image_metadata2["ImageId"][i] + ".png": image_metadata2["PredictedClass"][i]

                               for i in image_metadata2.index}



    output2 = pd.DataFrame.from_dict(all_images_target_class2, orient='index')

    output2.to_csv(

        r"C:\Users\avsanjay\PycharmProjects\kaggle\cleverhans-master\cleverhans-master\examples\nips17_adversarial_competition\sample_targeted_attacks\step_target_class\input\target_class2.csv")

            #if image_metadata.iloc[:,["ImageId"]] == newfilename:

                #image_metadata.iloc[["ImageId"== newfilename],["predicted_target"]] = predicted_targeted_classes[i]

                #i = i+1



        # all_images_target_class = {image_metadata["ImageId"][i] + ".png": image_metadata["TargetClass"][i]

        # for i in image_metadata.index}

        #with tf.Graph().as_default():

        #    x_input1 = tf.placeholder(tf.float32, shape=batch_shape)



        #    with slim.arg_scope(inception.inception_v3_arg_scope()):

        #        _, end_points = inception.inception_v3(x_input1, num_classes=num_classes, is_training=False)



            #predicted_labels = tf.argmax(end_points['Predictions'], 1)



        #    predicted_targeted_classes = sess.run(predicted_labels, feed_dict={x_input: targeted_images})



            #saver = tf.train.Saver(slim.get_model_variables())

            #session_creator = tf.train.ChiefSessionCreator(

            #    scaffold=tf.train.Scaffold(saver=saver),

            #    checkpoint_filename_with_path=checkpoint_path,

            #    master=tensorflow_master)



            #with tf.train.MonitoredSession(session_creator=session_creator) as sess:

        #predicted_targeted_classes = sess.run(predicted_labels, feed_dict={x_input: targeted_images})



            #predicted_targeted_classes_names = (pd.DataFrame({"CategoryId": predicted_targeted_classes})
