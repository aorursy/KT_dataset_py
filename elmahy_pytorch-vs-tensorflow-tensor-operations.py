"I will do more" if "you upvoted" else ""
import torch

import tensorflow as tf
tensor_torch = torch.ones((2,), dtype=torch.int8)



tensor_tf = tf.ones([3, 4], tf.int8) 
torch_tensor_methods = set([method_name for method_name in dir(tensor_torch)

                  if callable(getattr(tensor_torch, method_name)) and not method_name.startswith("_") and not method_name.endswith("_")])



tf_tensor_methods = []

for method_name in dir(tensor_tf):

    try:

        if callable(getattr(tensor_tf, method_name)) and not method_name.startswith("_") and not method_name.endswith("_"):

            tf_tensor_methods.append(method_name)

    except:

        # function that doesnt work in eager execution caused error

        pass

    

tf_tensor_methods = set(tf_tensor_methods)
# only two methods are common

common_methods = torch_tensor_methods.intersection(tf_tensor_methods)

common_methods
# Tf tensor has only few methods

# Math methods are sperate functions.

tf_tensor_methods.difference(torch_tensor_methods)
# In torch, tensors have the math methods with the tensor object 

print(torch_tensor_methods.difference(tf_tensor_methods))
# In tensorflow math functions, are separate functions that run on tensors

import sys

from inspect import getmembers, isfunction

tf_math_methods = set([o[0] for o in getmembers(tf.math, isfunction)])

tf_linalg_methods = set([o[0] for o in getmembers(tf.linalg, isfunction)])

tf_methods = tf_math_methods.union(tf_linalg_methods)
print(len(tf_methods.intersection(torch_tensor_methods)), 'of methods are the same name')

print(tf_methods.intersection(torch_tensor_methods))

same_methods = tf_methods.intersection(torch_tensor_methods)
# 11 tf methods start with reduce 

tf_methods_with_reduce = [i for i in tf_methods if i.startswith("reduce")]

print("{} tf methods start with reduce".format(len(tf_methods_with_reduce)))

print(tf_methods_with_reduce)





reduce_methods_in_torch = [i[7:] for i in tf_methods_with_reduce if  i[7:] in torch_tensor_methods ]

reduce_methods_same_in_tf_but_startswith_reduce = [i for i in tf_methods_with_reduce if  i[7:] in torch_tensor_methods ]

print("{} of them are the same with torch but not starting with reduce".format(len(reduce_methods_in_torch)))

print(reduce_methods_in_torch)



reduce_methods_not_in_torch = [i[7:] for i in tf_methods_with_reduce if  i[7:] not in torch_tensor_methods ]

print("Only {} of them are not in torch".format(len(reduce_methods_not_in_torch)))



print(reduce_methods_not_in_torch)
common_methods_union = common_methods.union(same_methods, reduce_methods_in_torch, reduce_methods_same_in_tf_but_startswith_reduce)
rest_of_torch_methods = torch_tensor_methods.difference(common_methods_union)

rest_of_tf_methods = tf_methods.difference(common_methods_union)
torch_startswith_is = set([i for i in rest_of_torch_methods if i.startswith('is')])

print("methods that starts with is in torch",torch_startswith_is)



tf_startswith_is = set( [i for i in rest_of_tf_methods if i.startswith('is')])

print("methods that starts with is in tf",tf_startswith_is)





methods_discussed_so_far = common_methods_union.union(torch_startswith_is, tf_startswith_is)
# Additional math functions in tensorflow 

print("Additional math functions in tensorflow")

print(','.join(tf_methods.difference(methods_discussed_so_far)))
print("Additional functions in torch tensor")

torch_methods_not_dicussed_yet = set(torch_tensor_methods.difference(methods_discussed_so_far))

','.join(torch_methods_not_dicussed_yet)

keras_K_function = set([o[0] for o in getmembers(tf.keras.backend, isfunction)])


common_between_torch_and_K = torch_methods_not_dicussed_yet.intersection(keras_K_function)

print("{} methods are common between torch and K".format(len(common_between_torch_and_K)))

print(','.join(common_between_torch_and_K))
diff1 = keras_K_function.difference(common_between_torch_and_K)

print(len(diff1))

K_methods_not_discusses_yet = diff1.difference(methods_discussed_so_far)

print('K_methods_not_discusses_yet are ',len(K_methods_not_discusses_yet))

torch_methods_not_discusses_yet = torch_methods_not_dicussed_yet.difference(common_between_torch_and_K)

print('torch_methods_not_discusses_yet are ',len(torch_methods_not_discusses_yet))



print('K_methods_not_discusses_yet')

','.join( K_methods_not_discusses_yet)
print('torch_methods_not_discusses_yet')

','.join( torch_methods_not_discusses_yet)


from tensorflow.python.ops import array_ops



set([o[0] for o in getmembers(array_ops, isfunction)])