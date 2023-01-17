# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt

# Uncomment to see where your variables get placed (see below)
# tf.debugging.set_log_device_placement(True)
tensor = tf.constant([[0,1,2],[3,4,5]])
var = tf.Variable(tensor)

# tf variables can be all types such as boolean or complex numbers
tf.Variable([False,False,True,True])

print("var = ", var)
print("shape = ", var.shape)
print("dtype = ", var.dtype)
print("numpi = ", var.numpy, "\n\n",np.array(var))

print("var = ", var)
print("\ntensor (constant) = ", tf.convert_to_tensor(var))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(var, ([1,6])))
# assignment is psossible with veriables as inline operation like list.append
var.assign([[18,1,2],[3,4,5]])
print(var)
# a and b are on the different locations of memory
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]

new_var = tf.Variable(a, name = "tekmen0")
print(new_var)
# this variable will not be differentiated nor updated
step_counter = tf.Variable(1, trainable=False)
#to create variables on cpu
with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)

#store variables in cpu but do the computation on the gpu
#however this is slower than doing all of the things on gpu
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)
x = tf.Variable(3.0)

# record operation to compute gradients
with tf.GradientTape() as tape:
  y = x**2

#get the gradient of y with respect to x
dy_dx = tape.gradient(y,x)

print(dy_dx)
print(dy_dx.numpy())
w = tf.Variable(tf.random.normal((3,2), name="w"))
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name="b")
x = [[1.,2.,3.]]

# persistent opsiyonu default olduğu gibi False olunca
# tape.gradient() methodu çağrılır çağırılmaz gradientler 
# kayboluyor memoryden, multiple output çağırırken önemli
with tf.GradientTape(persistent=True) as tape:
    #y tensor olarak kalıyor
    y = tf.matmul(x,w) + b
    loss = tf.reduce_mean(y)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(dl_dw, "\n\n",dl_db)

my_vars = { "w" : w,
              "b" : b}

#dictionaries are also possible 
grads = tape.gradient(loss, my_vars)
print(grads["w"])

# will release tape (it is released as defult from one 
# call after of gradient)
del tape
layer = tf.keras.layers.Dense(2, activation = "relu")
x = tf.constant([[1.,2.,3.]])

with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y)

grads = tape.gradient(loss, layer.trainable_variables)

# do not forget that layer.trainable_variables are just tensorflow variables
# (n_input, n_next_features) is the dimensions of the weights in tensorflow
for var, g in zip(layer.trainable_variables, grads):
    print(var.name, g.shape)
#bir modeldeki featurelere nasıl ulaşırız? ???
# bir modeldeki weights'e ve biase layer.trainable_variables ya da model.trainable_variables gibi bir özellikle erişebiliriz

# Trainable variable
x0 = tf.Variable(3.0, name="x0")
# Non-trainable variable 
x1 = tf.Variable(3.0, name="x1", trainable=False)
# Not a variable because variable + tensor returns tensor
x2 = tf.Variable(2.0, name="x2") + 1.0 #var + tensor
# Not a variable
x3 = tf.constant(3.0, name="x3")


# gradient tape sadece computationları tutar, gradients() methodu tutulan computation grapha göre hesaplama yapar
# ve persisten = False'de 1 kere hesaplama yaptı mı gradientler kaybolur.
with tf.GradientTape(persistent = True) as tape:
    y = (x0**2) + (x1**2) + (x2**2)
    
grad = tape.gradient(y,[x0,x1,x2,x3])

for g in grad:
    print(g)
    
# only will compute gradient for x0
# list the watched variables by tape
variables = [var for var in tape.watched_variables()]
for var in variables:
    print(var.name)
# to watch tensors and other constants we can use GradientTape().watch(tensor) utility
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2

dy_dy = tape.gradient(y,x)
print(dy_dx.numpy())
x0 = tf.Variable(0.0)
x1 = tf.Variable(1.)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y = tf.nn.softplus(x1 + x2)
    
grads = tape.gradient(y,[x1,x0])

#since x0 is not activated by default, no ggradient calculation run on it
for i in grads:
    print(i)
x = tf.Variable(3., name="x")

with tf.GradientTape() as tape:
    y = x**2
    z = 3*y

#it is possible to get intermediate value as grad, namely variable y, even if they are tensors
#they are not listed in the watched variables
    
for g in tape.watched_variables():
    print(g.name)
print()    

grad = tape.gradient(z, [y,x])

for g in grad:
    print(g)
# if gradient targets (y's) are multiple or not a scalar, 
# sum of gradients or gradient of sums are calculated

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
  y0 = x**2
  y1 = 1 / x

print(tape.gradient({'y0': y0, 'y1': y1}, x).numpy())

# it makes to get cost (sum of losses) easier
x = tf.Variable(2.)

with tf.GradientTape() as tape:
  y = x * [3., 4.]

print(tape.gradient(y, x).numpy())

# to get every values gradient seperate, there is jacobian option
# linear aralıklı tensor oluştur
x = tf.linspace(-10, 10 , 200+1)

with tf.GradientTape() as tape:
    tape.watch(x)
    # burada y'de her elemente göre y vektörünün gradientler toplamını alsa bile elementwise gradient almış oluyor
    # jacobian hesaplamayı skip edebiliriz bu yüzden her element bağımsız olduğu için
    y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y,x)

plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label="dy_dx")
plt.legend()
_ = plt.xlabel("x")
# if while gibi controlf-flow durumları doğal olarak handle eder tensorflow
# çünkü hesaplandıkça kaydediyor işlemi

x = tf.constant(1.0)

v0 = tf.Variable(2.)
v1 = tf.Variable(2.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    
    if x > 0.0:
        result = v0
    else:
        result = v1**2
    
dv0, dv1 = tape.gradient(result, [v0,v1])

print(dv0)
print(dv1)

#control flow operations are invisible to gradient based optimizers
# gradient returns None if variable and function is not connected
x = tf.Variable(1.0)
y = tf.Variable(2.0)

with tf.GradientTape() as tape:
    z = y**2

dy, dx = tape.gradient(z, [y,x])

print("gradient returns None if variable and function is not connected:\n")
print(dy)
print(dx)

# gradient returns None if variable operation is unintentionally returned tensor
# since tensors aren't watched by tape default

with tf.GradientTape() as tape:
    z = y + 1 #var + tensor

print("\n", "#"*100, "\n")
print("gradient returns None if variable operation is unintentionally returned tensor")
print(tape.gradient(z,y))

# gradient returns None when calculations are did outside tensorflow

x = tf.Variable(([[0,1],[2,3]]))
with tf.GradientTape() as tape:
    x2 = x**2

    #not a tensorflow operation
    y = np.mean(x2)

    y = tf.reduce_mean(y)

print("\n", "#"*100, "\n")
print("gradient returns None when calculations are did outside tensorflow")
print(tape.gradient(y,x))

# dtype = integer olan şeylerin graidentini alma genelde
# bazı gradient sonlandırıcı işlemlerde None dönüyor mesela x1.assign_add(x0), x0'ı hesaplamayacak
# bazı float işlemleri de gradient hesaplama implementasyonu yapılmamış işlemler, contrast ayarlama gibi



