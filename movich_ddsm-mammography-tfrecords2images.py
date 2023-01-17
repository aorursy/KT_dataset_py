!pip install tensorflow==1.15
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
# make sure everything was written properly by reading it back out

def read_and_decode_single_example(filenames):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)

    

    reader = tf.TFRecordReader()

    

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(

        serialized_example,

        features={

            'label_normal': tf.FixedLenFeature([], tf.int64),

            'image': tf.FixedLenFeature([], tf.string)

        })

    

    # now return the converted data

    label = features['label_normal']

    image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.reshape(image, [299, 299, 1])

    

    return label, image
label, image = read_and_decode_single_example(["../input/ddsm-mammography/training10_0/training10_0.tfrecords",

                                               "../input/ddsm-mammography/training10_1/training10_1.tfrecords",

                                              "../input/ddsm-mammography/training10_2/training10_2.tfrecords",

                                              "../input/ddsm-mammography/training10_3/training10_3.tfrecords",

                                              "../input/ddsm-mammography/training10_4/training10_4.tfrecords"])#,

images_batch, labels_batch = tf.train.batch([image, label], batch_size=16, capacity=56000)

global_step = tf.Variable(0, trainable=False)
import matplotlib

labs=[]

try:

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())



        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        for j in range(0,55891):

            la_b, im_b = sess.run([labels_batch, images_batch])

            for i in range(13):

                #plt.savefig(im_b[i]+str(".png"))

                #plt.imshow(im_b[i].reshape([299,299]))

                #plt.title("Label: " + str(la_b[i]))

                #plt.show()



                matplotlib.image.imsave(str("scan_")+str(j)+str(".png"),im_b[i].reshape([299,299]))

                labs.append(str(la_b[i]))



        coord.request_stop()



        # Wait for threads to stop

        coord.join(threads)

except tf.errors.OutOfRangeError:

    pass
with open('labels_DDSM_Mamography.txt', 'w') as f:

    for item in labs:

        f.write("%s\n" % item)