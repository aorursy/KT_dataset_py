import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
a = np.array([3, 2])
print("An array:")
print(a)
print("Shape of a: (note the weird shape)")
print(a.shape)
print("==================================")
x = np.array([[3, 2]])
print("A row vector:")
print(x)
print("Shape of x:")
print(x.shape)
print("==================================")
y = np.array([[3], [2]])
print("A column vector:")
print(y)
print("Shape of y:")
print(y.shape)
X = np.array([[3, 2], [-1, 3]])
print("This is a matrix")
print(X)
print("Shape of X:")
print(X.shape)
print(X.reshape(1,4))
print(X.reshape(1,-1)) # Notice the -1
a = np.arange(2,12)
a
a[3:8] # Extract a subarray from the 3rd element to (but not include) the 8th one.
my_string = "Hallo"
my_substring = my_string[1:4] # instead of using some string.substring() function, if it even exists!
print(my_substring)
a = np.array([[1, 8, -3, 0], [2, 1, -1, 5], [6, 5, 7, 1], [-2, -1, 1,4]])
print(a[1:3,1:3])
print(a[1:3,:]) # take all the columns in the sliced rows
print(a[:,1:3]) # take all the rows in the sliced columns
a = np.array([[1, 8, -3, 0], [2, 1, -1, 5], [6, 5, 7, 1], [-2, -1, 1,4]])
a = np.tile(a, (3,1,1)).T
a
plt.imshow(np.maximum(a/np.max(a), 0))
print(a[:,:,1]) # Get the green channel (every values for width and height)
a[:,:,1] = np.zeros((4,4)) # Set the green channel to all 0s
a
plt.imshow(np.maximum(a/np.max(a), 0)) # red + blue = pink
a = np.array([[1, 8, -3, 0], [2, 1, -1, 5], [6, 5, 7, 1], [-2, -1, 1,4]])
a = np.tile(a, (3,1,1)).T
a = np.tile(a, (50,1,1,1))
print(a[5,:,:,1]) # Get the green channel (every values for width and height) of the $5^{th}$ image
print(np.sum(a, axis=3).shape)
np.sum(a, axis=3)
print((np.sum(a, axis=(0))/50).shape)
np.sum(a, axis=(0))/50
np.sum(a, axis=(0,3))
a = np.array([[2, 1, -1, 0]]) # A row vector with the shape (1,4)
b = 2 # A scalar value
# What is a + b? Is it compatible to add a vector and a scalar?
c = a + b
print("a + b:")
print(c)
print(a.shape==c.shape)
brd_b = np.array([2, 2, 2, 2]) # brd_b = np.repeat(b, 4, axis=0)
print(brd_b)
print(c)
print(np.array_equal(a + b, a + brd_b))

X = np.arange(1,10).reshape(3,3)
print("X:")
print(X)
d = np.array([1,-1,1]).reshape(1,-1)  # Note how we reshaped d 
print("d:")
print(d)
print("X + d: (d is broadcasted along the first dimension)")
print(X + d) 
print("\n========================")
d = np.array([1,-1,1]).reshape(-1,1)  # Note how we reshaped d 
print("d:")
print(d)
print("X + d: (d is broadcasted along the second dimension)")
print(X + d)
a = np.arange(20, step=2)
print(a)
print("Min: " + str(a.min()))
print("Max: " + str(a.max()))
print("Mean: " + str(a.mean()))
print("Standard Deviation: " + str(a.std()))
print("Norm of vector a: " + str(np.linalg.norm(a)))

x = np.array([128, 1, -3, 2, 0.4, 1000]).reshape(-1,1)
x_prime = (x - x.min()) / (x.max() - x.min())
print(x_prime)
plt.bar(range(x.shape[0]), x_prime.ravel())
x = np.array([128, 1, -3, 2, 0.4, 1000]).reshape(-1,1)
x_std = (x - x.mean()) / x.std()
print(x_std)
plt.bar(range(x.shape[0]), x_std.ravel())
print(np.random.randn(2,4)) # You should use this for your weight initialization
print(np.random.rand(2,4))
print(np.random.randint(0,10)) # note the upper limit
print(np.random.permutation([1, -4, 3, 2, -6]))

arr = np.arange(9).reshape((3, 3))
print(np.random.permutation(arr))
print("When we do not specify the seed:")
for i in range(3):
    print(np.random.randint(0,10))
print("When we specify the same seed:")
for i in range(3):
    np.random.seed(1111) # set the same seed 1111 before every random generation
    print(np.random.randint(0,10))
print("When we specify different seeds:")
for i in range(3):
    np.random.seed(i * 3)
    print(np.random.randint(0,10))
n_in = 3 # the number of neurons from the previous layer
n_out = 2 # the number of neurons from the current layer
W = np.random.randn(n_out, n_in) * (0.1) 
print(W)
n_in = 3 # the number of neurons from the previous layer
n_out = 2 # the number of neurons from the current layer
W = np.random.randn(n_out, n_in) * (np.sqrt(2. / (n_in + n_out)))  # Xavier initialization
print(W)
n_in = 3 # the number of neurons from the previous layer
n_out = 2 # the number of neurons from the current layer
W = np.random.randn(n_out, n_in) * (np.sqrt(1. / n_in))  # He initialization
print(W)
def sm_sample(pa = 0.3, pb = 0.6, pc = 0.1):

    r = np.random.rand()
    if (r < pa):
        output = 'a'
    elif (r < pa + pb):
        output = 'b'
    else:
        output = 'c'
    return output

outputs = []
for i in range(10):
    outputs.append(sm_sample())
print(outputs)

outputs = []
# Law of large numbers: 100000 is large enough for our sample to approximate the true distribution
for i in range(100000):  
    outputs.append(sm_sample())

from collections import Counter
c_list = Counter(outputs)
print(c_list) 
def sm_sample_general(out, smp):
    # out contains possible outputs
    # smp contains the softmax output distributions
    return np.random.choice(out, p = smp)

out = ['a', 'b', 'c']
smp=np.array([0.3, 0.6, 0.1])

outputs = []
for i in range(10):
    outputs.append(sm_sample_general(out, smp))
print(outputs)

outputs = []
# Law of large numbers: 100000 is large enough for our sample to approximate the true distribution
for i in range(100000):  
    outputs.append(sm_sample_general(out, smp))

from collections import Counter
c_list = Counter(outputs)
print(c_list) 
# The normal for-loop
def my_dot(a, b):
    out = 0
    #assert(a.shape[0]==b.shape[0])
    for i in range(a.shape[0]):
        out += a[i] * b[i]
    return out
n = 1000
a = np.random.randn(n)
b = np.random.randn(n)

%%timeit -n 10
my_dot(a,b)
%%timeit -n 10
np.dot(a,b)
np.random.seed(1234)

dropout_p = 0.8                         # keep around 80% number of neurons

A_prev = np.random.randn(100, 50)       # Activations from the previous layer, mini-batch training
W = np.random.randn(30, 100)            # the previous layer has 100 neuron, 
                                        # the current layer to be dropped-out has 30 neurons
b = np.random.randn(30, 1)

# Dropout
# CODE HERE                             # Initialize a random, uniform-distribute vector with the same shape
# CODE HERE                             # Mask the drop-out: True (1) if we want to keep the corresponding neuron
                                        # False (0) if the neuron is dropped out
# CODE HERE                             # Apply the mask
# CODE HERE                             # Scale the activation after dropping a number of neurons

# Do the forward pass
A = np.tanh(np.dot(W, A_prev) + b)

print(A[3:6, 2:4])

np.random.seed(1234)

dropout_p = 0.8                         # keep around 80% number of neurons

A_prev = np.random.randn(100, 50)       # Activations from the previous layer, mini-batch training
W = np.random.randn(30, 100)            # the previous layer has 100 neuron, 
                                        # the current layer to be dropped-out has 30 neurons
b = np.random.randn(30, 1)

# Dropout
dropped = np.random.rand(100, 50)       # Initialize a random, uniform-distribute vector with the same shape
masked = (dropped < dropout_p)          # Mask the drop-out: True (1) if we want to keep the corresponding neuron
                                        # False (0) if the neuron is dropped out
A_prev = A_prev * masked                # Apply the mask
A_prev = A_prev / dropout_p             # Scale the activation after dropping a number of neurons

# Do the forward pass
A = np.tanh(np.dot(W, A_prev) + b)

print(A[3:6, 2:4])

