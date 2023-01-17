import tensorflow as tf
# Tensor is a multi-dimensional array

# Tensor with zero dimension (axis) => Scalar

# Tensor with one dimension (axis) => Vector

# Tensor with two dimension (axis) => Matrix

# Tensor with more than two dimension (axis) => Tensor



# create a tensor with one dimension

t = tf.range(12)

t
# inspect the shape of tensor

t.shape
# inspect number of elements in tensor = product of all the shape elements

tf.size(t)
# change the shape of tensor => convert into two dimension

tf.reshape(t, (2, 6))
# change the shape of tensor => convert into two dimension

# but not give all shape element => tensor will calculate automatically

tf.reshape(t, (-1, 6))
# change the shape of tensor => convert into two dimension

# but not give all shape element => tensor will calculate automatically

tf.reshape(t, (2, -1))
# create a tensor with three dimension and zero value initialized

tf.zeros((2, 3, 4))
# create a tensor with three dimension and one value initialized

tf.ones((2, 3, 4))
# create a tensor with three dimension and random value initialized

# random value are sampled from stadard normal distribution

tf.random.normal((2, 3, 4))
# create a tensor with two dimension and value we give

tf.constant([[1, 2, 3, 4], [2, 3, 4, 5]])
# element-wise mathematical operation

t1 = tf.constant([1, 2, 3, 4, 5])

t2 = tf.constant([2, 3, 4, 5, 6])

t3 = tf.constant([1.0, 2.0, 3.0])



t1+t2, t1-t2, t1*t2, t1/t2, t1**t2
# exponential value for each element in tensor

tf.exp(t3)
# concatenate two tensor

t1 = tf.ones((3, 4))

t2 = tf.zeros((3, 4))



tf.concat([t1, t2], axis=0), tf.concat([t1, t2], axis=1)
# element-wise logical operation for two tensor

t1==t2, t1>t2, t1<t2
# summation of all elements in tensor

t = tf.reshape(tf.range(15), (3, -1))

t, tf.reduce_sum(t)
# access element in tensor with index

t = tf.reshape(tf.range(15), (5, -1))

t, t[0, :], t[:, 0]
# tensor is immutable in tensorflow

# however, "Variable" tensor is an exception

t = tf.random.normal((3, 4))

t
t[0, 1] = 0.45
var_t = tf.Variable(t)

var_t
var_t[0, 2].assign(100)
var_t[2, :].assign(tf.range(4, dtype=tf.float32))
# running operation will cause new memory to be allocated to host result

x = tf.range(5, dtype=tf.float32)

y = tf.ones((5))

x, y
# a new memory space is allocated to host result

# model will have tens of thousands of parameters which should be updated in place

address = id(y)

y = x + y

id(y), address
var_z = tf.Variable(tf.zeros((5)))

var_z
address = id(var_z)

var_z.assign(x+y)

var_z
id(var_z)==address
tensor = tf.constant([1, 2, 3])

tensor
# tensor to numpy array

arr = tensor.numpy()

arr
# numpy array to tensor

tensor = tf.constant(arr)

tensor
# tenosr without dimension

scalar_tensor = tf.constant(3)

scalar_tensor
# tensor to scalar

scalar_tensor.numpy(), int(scalar_tensor), float(scalar_tensor)