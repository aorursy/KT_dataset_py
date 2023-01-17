import tensorflow as tf

def _transformer(x): return x, x
dataset = tf.data.Dataset.range(10).map(_transformer)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run([x, y]))
