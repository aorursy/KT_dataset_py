import tensorflow as tf

import numpy as np

import torch



print(tf.__version__)

print(np.__version__)

print(torch.__version__)
# Numpy 

print()

print("""Numpy Implementation""")

a = np.array(10)

print(a)

print(a.shape, a.dtype) # shape of the array and type of the elements

print()

a = np.array([10])

print(a)

print(a.shape, a.dtype) # shape of the array and type of the elements

print()

a = np.array([10], dtype=np.float32)

print(a)

print(a.shape, a.dtype) # shape of the array and type of the elements

print()



# TensorFlow

print("""Tensorflow Implementation""")

print()

b = tf.constant(10) # As Scalar

print(b)

print()

b = tf.constant(10, shape=(1,1)) # As 1-D Vector

print(b)

print()

b = tf.constant(10, shape=(1,1), dtype=tf.float32) # As 1-D Vector with specified Data-type

print(b)



# Torch

print()

print("""Torch Implementation""")

print()

c = torch.tensor(10, ) # As Scalar

print(c)

print()

c = torch.tensor([10]) # As 1-D Vector

print(c, c.shape, c.dtype)

print()

c = torch.tensor([10], dtype=torch.float32) # As 1-D Vector with specified Data-type

print(c)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1,2,3], [4,5,6]])

print(a)

print(a.shape, a.dtype)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.constant([[1,2,3], [4,5,6]])

print(b)

print(b.shape)

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor([[1,2,3], [4,5,6]]) 

print(c)

print(c.shape)
# Numpy

print()

print("""Numpy Implementation""")

a = np.zeros((3,3))

print(a, a.shape, a.dtype)

print()

a = np.ones((3,3))

print(a, a.shape, a.dtype)

print()

a = np.eye(3)

print(a, a.shape, a.dtype)

print()

a = np.full((3,3),10.0)

print(a, a.shape, a.dtype)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.zeros((3,3))

print(b)

print()

b = tf.ones((3,3))

print(b)

print()

b = tf.eye(3)

print(b)

print()

b = tf.fill([3,3], 10)

print(b)

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.zeros((3,3))

print(c)

print()

c = torch.ones((3,3))

print(c)

print()

c = torch.eye(3)

print(c)

print()

c = c.new_full([3,3], 10)

print(c)
# Numpy

print()

print("""Numpy Implementation""")

a = np.random.randn(3,3) 

print(a, a.shape, a.dtype)

print()

print(a.mean(), a.std())

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.random.normal((3,3),mean=0, stddev=1)

print(b)

print()

print(tf.reduce_mean(b), tf.math.reduce_std(b))

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.normal(mean=0, std=1, size=(3, 3))

print(c)

print()

print(torch.mean(c), torch.std(c))
# Numpy

print()

print("""Numpy Implementation""")

a = np.random.uniform(low=0, high=1, size=(3,3)) 

print(a, a.shape, a.dtype)

print()

print(a.mean(), a.std())

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.random.uniform((3,3), minval=0, maxval=1) # Values are always > 1

print(b)

print()

print(tf.reduce_mean(b), tf.math.reduce_std(b))

print()



# Torch

print()

print("""Torch Implementation""")

num_samples = 3

Dim = 3

c = torch.distributions.Uniform(0, +1).sample((num_samples, Dim))

print(c)

print()

print(torch.mean(c), torch.std(c))
# Numpy

print()

print("""Numpy Implementation""")

a = np.arange(0,9)

print(a)

print()

a = np.arange(start=1, stop=20, step=2, dtype=np.float32)

print(a, a.dtype)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.range(9)

print(b)

print()

b = tf.range(start=1, limit=20, delta=2, dtype=tf.float64)

print(b)

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.arange(start=0, end=9)

print(c)

print()

c = torch.arange(start=1, end=20, step=2, dtype=torch.float64)

print(c)

# Numpy

print()

print("""Numpy Implementation""")

a = a.astype(np.uint8)

print(a, a.dtype)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.cast(b, dtype=tf.uint8)

print(b)

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor(c)

c= c.type(torch.int64)

print(c)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([1,2,3,4,5])

b = np.array([6,7,8,9,10])

c = np.add(a, b) # x + y

print(c, c.dtype)

print()

c = np.subtract(b,a) # y - x

print(c, c.dtype)

print()

c = np.divide(b,a) # y / x

print(c, c.dtype)

print()

c = np.multiply(b,a) # y * x

print(c, c.dtype)

print()

c = (a **2)

print(c)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

x = tf.constant([1,2,3,4,5])

y = tf.constant([6,7,8,9,10])

z = tf.add(x,y) # x + y

print(z)

print()

z = tf.subtract(y,x) # y - x

print(z)

print()

z = tf.divide(y,x) # y / x

print(z)

print()

z = tf.multiply(y,x) # y * x

print(z)

print()

z = (x **2)

print(z)

print()



# Torch

print()

print("""Torch Implementation""")

t = torch.tensor([1,2,3,4,5])

u = torch.tensor([6,7,8,9,10])

v = torch.add(t, u) # x + y

print(v)

print()

v = torch.sub(u,t) # y - x

print(v)

print()

v = torch.true_divide (u, t) # y / x

print(v)

print()

v = torch.mul(u,t) # y * x

print(v)

print()

v = (t **2)

print(v)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([1,2,3,4,5])

b = np.array([6,7,8,9,10])

c = np.dot(a, b) # x + y

print(c, c.dtype)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

x = tf.constant([1,2,3,4,5])

y = tf.constant([6,7,8,9,10])

z = tf.tensordot(x,y, axes=1)

print(z)

print()



# Torch

print()

print("""Torch Implementation""")

t = torch.tensor([1,2,3,4,5])

u = torch.tensor([6,7,8,9,10])

v = torch.dot(t,u)

print(v)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1,2,3], [4,5,6]])

b = np.array([[1,2,3], [4,5,6], [7,8,9]])

c = np.matmul(a,b) # (2,3) @ (3,3) --> (2,3) output shape

print(c)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

x = tf.constant([[1,2,3], [4,5,6]])

y = tf.constant([[1,2,3], [4,5,6], [7,8,9]])

z = tf.matmul(x,y) # (2,3) @ (3,3) --> (2,3) output shape

print(z)

print()



# Torch

print()

print("""Torch Implementation""")

t = torch.tensor([[1,2,3], [4,5,6]])

u = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])

v = torch.matmul(t,u) # (2,3) @ (3,3) --> (2,3) output shape

print(v)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([1,2,3,4,5,6,7,8])

print(a[:])

print(a[2:-3])

print(a[3:-1])

print(a[::2])

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.constant([1,2,3,4,5,6,7,8])

print(b[:])

print(b[2:-3])

print(b[3:-1])

print(b[::2])

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor([1,2,3,4,5,6,7,8])

print(c[:])

print(c[2:-3])

print(c[3:-1])

print(c[::2])
# Numpy

print()

print("""Numpy Implementation""")

indices = np.array([0,3,5])

x_indices = a[indices]

print(x_indices)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

indices = tf.constant([0,3,5])

x_indices = tf.gather(b, indices)

print(x_indices)

print()



# Torch

print()

print("""Torch Implementation""")

indices = torch.tensor([0,3,5])

x_indices = c[indices]

print(x_indices)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1,2,3],

              [4,5,6],

              [7,8,9]])



# Matrix Indexing

# Print all individual Rows and Columns

print("Row-1",a[0, :])

print("Row-2",a[1, :])

print("Row-3",a[2, :])

print()

print("Col-1",a[:, 0])

print("Col-2",a[:, 1])

print("Col-3",a[:, 2])

print()



# Print the sub-diagonal matrix

print("Upper-Left",a[0:2,0:2])

print("Upper-Right",a[0:2,1:3])

print()

print("Bottom-Left",a[1:3,0:2])

print("Bottom-Right",a[1:3,1:3])





# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.constant([[1,2,3],

                 [4,5,6],

                 [7,8,9]])



# Matrix Indexing

# Print all individual Rows and Columns

print("Row-1",b[0, :])

print("Row-2",b[1, :])

print("Row-3",b[2, :])

print()

print("Col-1",b[:, 0])

print("Col-2",b[:, 1])

print("Col-3",b[:, 2])

print()



# Print the sub-diagonal matrix

print("Upper-Left",b[0:2,0:2])

print("Upper-Right",b[0:2,1:3])

print()

print("Bottom-Left",b[1:3,0:2])

print("Bottom-Right",b[1:3,1:3])



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor([[1,2,3],

                 [4,5,6],

                 [7,8,9]])



# Matrix Indexing

# Print all individual Rows and Columns

print("Row-1",c[0, :])

print("Row-2",c[1, :])

print("Row-3",c[2, :])

print()

print("Col-1",c[:, 0])

print("Col-2",c[:, 1])

print("Col-3",c[:, 2])

print()



# Print the sub-diagonal matrix

print("Upper-Left",c[0:2,0:2])

print("Upper-Right",c[0:2,1:3])

print()

print("Bottom-Left",c[1:3,0:2])

print("Bottom-Right",c[1:3,1:3])
# Numpy

print()

print("""Numpy Implementation""")

a = np.arange(9)

print(a)

a = np.reshape(a, (3,3))

print(a)

a = np.transpose(a, (1,0)) # Swap axes (1,0), use (0,1) nothing happens

print(a)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.range(9)

print(b)

b = tf.reshape(b, (3,3))

print(b)

b = tf.transpose(b, perm=[1,0]) # Swap axes in perm (1,0), use (0,1) nothing happens

print(b)

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.arange(9)

print(c)

c = torch.reshape(c, (3,3))

print(c)

c = c.permute(1,0) # Swap axes in perm (1,0), use (0,1) nothing happens

print(c)
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1, 2], [3, 4]])

print("a",a)

b = np.array([[5, 6]])

print("b",b)

print()

d = np.concatenate((a, b), axis=0)

print("Concat (axis=0 - Row)")

print(d)

print()

e = np.concatenate((a, b.T), axis=1)

print("Concat (axis=1 - Column)")

print(e)

print()

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

x = tf.constant([[1, 2], [3, 4]])

print("x",x)

y = tf.constant([[5, 6]])

print("y",y)

print()

z = tf.concat((x, y), axis=0)

print("Concat (axis=0 - Row)")

print(z)

print()

z = tf.concat((x, tf.transpose(y)), axis=1)

print("Concat (axis=1 - Column)")

print(z)

print()

print()



# Torch

print()

print("""Torch Implementation""")

t = torch.tensor([[1, 2], [3, 4]])

print("x",t)

u = torch.tensor([[5, 6]])

print("y",u)

print()

v = torch.cat((t , u), axis=0)

print("Concat (axis=0 - Row)")

print(v)

print()

v = torch.cat((t , u.T), axis=1)

print("Concat (axis=1 - Column)")

print(v)

print()

# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1,2,3,4,5], [10,10,10,10,10]])

print(a)

print()

print("Overall flattened Sum", a.sum())

print("Sum across Columns",a.sum(axis=0)) 

print("Sum across Rows",a.sum(axis=1))

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.constant([[1,2,3,4,5], [10,10,10,10,10]])

print(b)

print()

print("Overall flattened Sum",tf.math.reduce_sum(b))

print("Sum across Columns",tf.math.reduce_sum(b, axis=0))

print("Sum across Rows",tf.math.reduce_sum(b, axis=1))

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor([[1,2,3,4,5], [10,10,10,10,10]])

print(c)

print()

print("Overall flattened  Sum",torch.sum(c))

print("Sum across Columns",torch.sum(c, axis=0))

print("Sum across Rows",torch.sum(c, axis=1))

print()
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[1,2,3,4,5], [10,10,10,10,10]])

print(a)

print()

print("Overall flattened mean", a.mean())

print("Sum across Columns",a.mean(axis=0)) 

print("Sum across Rows",a.mean(axis=1))

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

b = tf.constant([[1,2,3,4,5], [10,10,10,10,10]])

print(b)

print()

print("Overall flattened mean",tf.math.reduce_mean(b))

print("Sum across Columns",tf.math.reduce_mean(b, axis=0))

print("Sum across Rows",tf.math.reduce_mean(b, axis=1))

print()



# Torch

print()

print("""Torch Implementation""")

c = torch.tensor([[1,2,3,4,5], [10,10,10,10,10]], dtype=torch.float32)

print(c)

print()

print("Overall flattened mean",torch.mean(c))

print("Sum across Columns",torch.mean(c, axis=0))

print("Sum across Rows",torch.mean(c, axis=1))
# Numpy

print()

print("""Numpy Implementation""")

a = np.full((3,3),10.0)

print(a)

print(a.shape)

a = np.expand_dims(a, axis=0)

print(a)

print(a.shape)

b = np.full((3,3),20.0)

print(b)

b = np.expand_dims(b, axis=0)

print(b.shape)

c = np.concatenate((a,b), axis=0)

c = np.moveaxis(c,2,0) # Move 2nd dimension to 0th position

print(c)

print(c.shape)

print()



# Tensorflow

print()

print("""Tensorflow Implementation""")

x = tf.fill((3,3),10.0)

print(x)

print(x.shape)

x = tf.expand_dims(x, axis=0)

print(x.shape)

y = tf.fill((3,3),20.0)

print(y)

print(y.shape)

y = tf.expand_dims(y, axis=0)

print(y.shape)

z = tf.concat((x,y), axis=0)

z = tf.transpose(z, [1, 0, 2])

print(z.shape)

print()



# Torch

print()

print("""Torch Implementation""")

m1 = torch.ones((2,), dtype=torch.int32)

m1 = m1.new_full((3, 3), 10)

m1 = torch.unsqueeze(m1, axis=0)

print(m1)

print(m1.shape)

m2 = torch.ones((2,), dtype=torch.int32)

m2 = m2.new_full((3, 3), 20)

print(m2)

m2 = torch.unsqueeze(m2, axis=0)

print(m2.shape)

m = torch.cat((m1,m2), axis=0)

m = m.permute([1,0,2])

print(m)

m.shape
# Numpy

print()

print("""Numpy Implementation""")

a = np.array([[5,10,15],

               [20,25,30]])

b = np.array([[6,69,35],

              [70,10,82]])

c = np.array([[25,45,48],

             [4,100,89]])

print(a)

final = np.zeros((3,2,3))

print(final.shape)

final[0, :, :] = a

final[1, :, :] = b

final[2, :, :] = c

print(final)

print("Overall flattened max", final.max())

print("max across Columns",final.max(axis=0)) 

print("max across Rows",final.max(axis=1))

print()

print("Index of max value across the flattened max", final.argmax())

print("Index of max value across Columns",final.argmax(axis=0)) 

print("Index of max value across Rows",final.argmax(axis=1)) 



# Tensorflow

print()

print("""Tensorflow Implementation""")

final = tf.constant([[[5,10,15],

                  [20,25,30]],

                 [[6,69,35],

                  [70,10,82]],

                 [[25,45,48],

                  [4,100,89]]])

print(final)



print("Overall flattened max", tf.math.reduce_max(final))

print("max across Columns",tf.math.reduce_max(final,axis=0)) 

print("max across Rows",tf.math.reduce_max(final, axis=1))

print()

print("Index of max value across the flattened max", tf.math.argmax(final))

print("Index of max value across Columns",tf.math.argmax(final, axis=0)) 

print("Index of max value across Rows",tf.math.argmax(final, axis=1)) 



# Torch

print()

print("""Torch Implementation""")

final = torch.tensor([[[5,10,15],

                       [20,25,30]],

                      [[6,69,35],

                       [70,10,82]],

                      [[25,45,48],

                        [4,100,89]]])

print(final)



print("Overall flattened max", torch.max(final))

print("max across Columns",torch.max(final,axis=0)) 

print("max across Rows",torch.max(final, axis=1))

print()

print("Index of max value across the flattened max", torch.argmax(final))

print("Index of max value across Columns",torch.argmax(final, axis=0)) 

print("Index of max value across Rows",torch.argmax(final, axis=1)) 

# Numpy

print("""Numpy Implementation""")

a = np.array([[10,10,10],[10,10,10],

              [10,10,10],[10,10,10]])



b = np.array([[20,20,20],[20,20,20],

              [20,20,20],[20,20,20]])



c = np.array([[30,30,30],[30,30,30],

              [30,30,30],[30,30,30]])

final = np.zeros((3,4,3))

final[0, :, :] = a

final[1, :, :] = b

final[2, :, :] = c



print("Upper-Left",final[:, 0:2, 0:2])

print("Lower-Right",final[:, 2:, 1:])

print("Middle Elements",final[:,1:3, 1])



# Ignore Middle Elements

a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

b = np.array([[13,14,15],[16,17,18],[19,20,21],[22,23,24]])

c = np.array([[25,26,27],[28,29,30],[31,32,33],[34,35,36]])



final = np.zeros((3,4,3))

print(final.shape)

final[0, :, :] = a

final[1, :, :] = b

final[2, :, :] = c

# Though may work,but not efficient

print("Ignore Middle",final[:,[0,0,0,1,1,2,2,3,3,3], [0,1,2,0,2,0,2,0,1,2]])
# Tensorflow

print("""Tensorflow Implementation""")

final = tf.constant([[[10,10,10],[10,10,10],

                  [10,10,10],[10,10,10]],



                  [[20,20,20],[20,20,20],

                  [20,20,20],[20,20,20]],



                  [[30,30,30],[30,30,30],

                  [30,30,30],[30,30,30]]])



print("Upper-Left",final[:, 0:2, 0:2])

print("Lower-Right",final[:, 2:, 1:])

print("Middle Elements",final[:,1:3, 1])
# Torch

print("""Torch Implementation""")

a = torch.Tensor([[10,10,10],[10,10,10],

              [10,10,10],[10,10,10]])



b = torch.Tensor([[20,20,20],[20,20,20],

              [20,20,20],[20,20,20]])



c = torch.Tensor([[30,30,30],[30,30,30],

              [30,30,30],[30,30,30]])

final = np.zeros((3,4,3))

final[0, :, :] = a

final[1, :, :] = b

final[2, :, :] = c



print("Upper-Left",final[:, 0:2, 0:2])

print("Lower-Right",final[:, 2:, 1:])

print("Middle Elements",final[:,1:3, 1])



# Ignore Middle Elements

a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

b = torch.Tensor([[13,14,15],[16,17,18],[19,20,21],[22,23,24]])

c = torch.Tensor([[25,26,27],[28,29,30],[31,32,33],[34,35,36]])



final = np.zeros((3,4,3))

print(final.shape)

final[0, :, :] = a

final[1, :, :] = b

final[2, :, :] = c

# Though may work,but not efficient

print("Ignore Middle",final[:,[0,0,0,1,1,2,2,3,3,3], [0,1,2,0,2,0,2,0,1,2]])

print("That's it")