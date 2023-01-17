from tensorflow.examples.tutorials.mnist import input_data



import tensorflow as tf



flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')



mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)



sess = tf.InteractiveSession()

# Train

tf.initialize_all_variables().run()



batch_xs, batch_ys = mnist.train.next_batch(10)



print(batch_xs)

print(batch_ys)