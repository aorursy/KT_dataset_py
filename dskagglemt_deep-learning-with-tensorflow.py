# %tensorflow_version 2.x
import tensorflow as tf
tf.__version__
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
# sess = tf.Session() # deprecated

# sess = tf.compat.v1.Session()

# tf.compat.v1.Session(
#     target='', graph=None, config=None
# )

# Launch the graph in a session.
# sess = tf.compat.v1.Session()

# with tf.compat.v1.Session() as sess:

    # Evaluate the tensor `c`.
#     c = node1 * node2
#     print(sess.run(c)) 
#     print(sess.run([node1, node2]))

#     sess.close()
    
# print(sess.run([node1, node2]))
# sess.close()
g = tf.Graph()
with g.as_default():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    print(node1, node2)
# with tf.Session(graph=g) as sess:
with tf.compat.v1.Session(graph = g) as sess:    
    c = node1 * node2
    print(sess.run(c)) 
    print(sess.run([node1, node2]))
    
    sess.close()
g1 = tf.Graph()
with g1.as_default():
    a = tf.constant(5)
    b = tf.constant(2)
    c = tf.constant(3)
    
    d = tf.multiply(a,b)
    e = tf.add(c,b)
    f = tf.subtract(d,e)
sess = tf.compat.v1.Session(graph = g1)
outs = sess.run(f)
print(outs)

sess.close()
with tf.compat.v1.Session(graph = g1) as sess:    
    outs = sess.run(f)
    print(outs)
    
    sess.close()
# a = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32)
# b = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32)

# adder_node = a + b
