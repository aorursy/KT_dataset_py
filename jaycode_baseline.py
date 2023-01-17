import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["../input/train.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.

record_defaults = [[''], [''], [''], [''], [''], [''], [''], [0.0], [0.0]]
dates, category, descript, dayofweek, pddistrict, resolution, address, x, y = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.pack([x, y])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        example = sess.run([features, col1])

    coord.request_stop()
    coord.join(threads)
