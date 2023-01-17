import numpy as np # linear algebra
x = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
)

y = np.array(
    [
        [0],
        [1],
        [1],
        [0]
    ]
)
print(x)
print(y)
num_epochs = 60000

np.random.seed(1)
weight0 = 2*np.random.random((3, 4)) - 1
weight1 = 2*np.random.random((4, 1)) - 1
print(weight0)
print(weight1)
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
for j in range(num_epochs):
    k0 = x
    k1 = nonlin(np.dot(k0, weight0))
    k2 = nonlin(np.dot(k1, weight1))
    
    #how much did we miss the target value?
    k2_error = y - k2
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(k2_error))))
    
    #in what direction is the target value?
    k2_delta = k2_error*nonlin(k2, deriv=True)
    
    #how much did each k1 value contribute to k2 error
    k1_error = k2_delta.dot(weight1.T)
    
    k1_delta= k1_error * nonlin(k1,deriv=True)
    
    weight1 += k1.T.dot(k2_delta)
    weight0 += k0.T.dot(k1_delta)