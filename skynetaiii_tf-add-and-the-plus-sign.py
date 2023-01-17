import tensorflow as tf
tf.enable_eager_execution()

a = tf.constant([[1,2,3],[4,5,6]])
b = tf.constant([[6,5,4],[3,2,1]])
c = a+b
print(c)
print(tf.add(a,b))
