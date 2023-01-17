import tensorflow as tf
graph1 = tf.Graph()
with graph1.as_default():

    a = tf.constant([2], name = 'constant_a')

    b = tf.constant([3], name = 'constant_b')
print(a)
# Printing the value of a

# sess = tf.Session(graph = graph1) # Decomission

sess = tf.compat.v1.Session(graph = graph1)

result = sess.run(a)

print(result)

sess.close()
with graph1.as_default():

    c = tf.add(a, b)

    #c = a + b is also a way to define the sum of the terms
sess = tf.compat.v1.Session(graph = graph1)
result = sess.run(c)

print(result)
sess.close()
with tf.compat.v1.Session(graph = graph1) as sess:

    result = sess.run(c)

    print(result)
graph2 = tf.Graph()



with graph2.as_default():

    Scalar = tf.constant(2)

    Vector = tf.constant([5,6,2])

    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])

    Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )

    

with tf.compat.v1.Session(graph = graph2) as sess:

    result = sess.run(Scalar)

    print ("Scalar (1 entry):\n %s \n" % result)

    result = sess.run(Vector)

    print ("Vector (3 entries) :\n %s \n" % result)

    result = sess.run(Matrix)

    print ("Matrix (3x3 entries):\n %s \n" % result)

    result = sess.run(Tensor)

    print ("Tensor (3x3x3 entries) :\n %s \n" % result)
Scalar.shape
Tensor.shape
v = tf.Variable(0)
# update = tf.assign(v, v+1) # Depricated in TF 2.0

update = tf.compat.v1.assign(v, v+1)
# init_op = tf.global_variables_initializer() # Depricated in TF 2.0

init_op = tf.compat.v1.global_variables_initializer()
# with tf.compat.v1.Session(graph = graph1) as session:

#     session.run(init_op)

#     print(session.run(v))

#     for _ in range(3):

#         session.run(update)

#         print(session.run(v))
# a = tf.placeholder(tf.float32)