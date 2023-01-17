# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Tensors are immutable data types like python strings
# 0 dimensional tesnor with rank 0 (i.e scalar)
some_scalar = tf.constant(4) #no shape
print(some_scalar)
rank1_tensor = tf.constant([1.0,2.4,2.8])
print(rank1_tensor)
a = np.arange(6).reshape(2,-1)
rank2_tensor = tf.constant(a, dtype=tf.float64)
print(rank2_tensor)
arr = np.array(rank2_tensor)
print(arr)
new_rank2_tensor = tf.constant(np.ones((2,3)))*2 # or tf.ones([2,2])
print(new_rank2_tensor)
vector_rank2 = tf.ones([3,1], dtype=tf.float64)
print(tf.add(new_rank2_tensor, rank2_tensor))
print()
print(tf.matmul(rank2_tensor,vector_rank2))
print("tf.reduce_max(rank2_tensor) = ", tf.reduce_max(rank2_tensor))
arr = np.array(rank2_tensor)
arr[0,1] = 10
rank2_tensor = arr
# columlarda en büyüğü alıyor
print("tf.argmax(rank2_tensor) = ", tf.argmax(rank2_tensor)) #index olarak tensor dönmesi gerekirken [1 1 1] lik bir tensor dönrü
print()
print("tf.nn.softmax(rank2_tensor) = \n", tf.nn.softmax(rank2_tensor)) # gets softmax through axis 0
rank4_tensor = tf.zeros([2,3,4,5])

print("elements type\t=", rank4_tensor.dtype)
print("rank\t\t=", rank4_tensor.ndim)
print("shape\t\t=", rank4_tensor.shape)
print("\nelements along axis0  =", rank4_tensor[0])
print("\nelements of the last axis  =", rank4_tensor[-1])
print("num of elements (2*3*4*5) = ", tf.size(rank4_tensor).numpy()) 
#last dimesions are contiguous chunks in the memory
print("3rd dimension first 3\t=", rank4_tensor[1,0,:3,2])
dummy_tensor = tf.constant(np.arange(6).reshape(2,3))
print("First item in last column:", dummy_tensor[0, -1].numpy())
var_1 = tf.Variable(tf.constant([[3],[1],[4]]))
var_1 = tf.reshape(var_1,(1,-1)) # -1 i destekliyor
print(var_1.shape)
var_1 = tf.reshape(var_1,(3)),
print("\n var1 1rank= ", var_1, var_1)
# Bad examples: don't do this

rank_3_tensor = tf.Variable(tf.constant(np.arange(30).reshape(3,2,5)))
print(rank_3_tensor, "\n")
#good reshapes
print("\nGood Reshapes\n")

print(tf.reshape(rank_3_tensor, (3,2*5)), "\n")

print(tf.reshape(rank_3_tensor, (3*2,5)))


#bad reshapes
print("\nbad reshapes : \n")
# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

x = tf.constant(np.arange(6).reshape((2,3)), dtype=tf.int32)
v = tf.ones([2,1], dtype=tf.int32)*2 #rankın 2 olması şart

print("x =", x, "\n\nv = ", v, "\n\n x*v = ",x*v )
print(tf.broadcast_to(tf.constant([1,2,3]), [2,3]))
# kendi datatypemızı tensora çevirme utilityleri var
# string tensorları var
# sparse tensorlar var (boş yerlerde 0 yerine boşluk var)
# ragged tensorlar var (non-rectangular (ilk satır 3 uzunlukta, 2. satır 2, 3. satır 5 gibi)) (bu da bir sparse tensor)