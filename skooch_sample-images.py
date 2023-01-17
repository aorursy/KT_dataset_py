import tensorflow as tf
import matplotlib.pyplot as plt
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
label, image = read_and_decode_single_example(["../input/training10_0/training10_0.tfrecords", "../input/training10_1/training10_1.tfrecords"])
images_batch, labels_batch = tf.train.batch([image, label], batch_size=16, capacity=2000)
global_step = tf.Variable(0, trainable=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for j in range(3):
        la_b, im_b = sess.run([labels_batch, images_batch])
        
        for i in range(8):
            plt.imshow(im_b[i].reshape([299,299]))
            plt.title("Label: " + str(la_b[i]))
            plt.show()
            
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)