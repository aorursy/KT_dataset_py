import numpy as np

# Back-Propagation



# alpha is the Learning rate

alpha = 0.9

    

#x

x = np.array([[0,0,1],

              [0,1,1],

              [1,0,1],

              [1,1,1]])



#Correct output

d = np.array([[0,1,1,0]]).T


np.random.seed(1)

#Initialize the weights

synaptic_weights1 = 2 * np.random.random((3,4)) - 1

synaptic_weights2 = 2 * np.random.random((4,1)) - 1

print('Random starting synaptic weights1:')

print(synaptic_weights1)

print('Random starting synaptic weights2:')

print(synaptic_weights2)
#Training algorithm

for iteration in range(1):

    

    v1 = np.dot(x, synaptic_weights1)

    y1 = np.multiply(-1, 1+np.exp(-v1))   

    v = np.dot(synaptic_weights2.T, y1)

    y = np.multiply(-1, 1+np.exp(-v))

    print(y.shape)



    #Backpropagation

    e = d - y 

    print(e.shape)

    s = np.multiply(y,(1-y))

    Delta = np.multiply(s, e)

    print(Delta.shape)



    e1 = np.dot(synaptic_weights2.T, Delta)

    s1 = np.multiply(y1,(1-y1))

    Delta1 = np.multiply(s1, e1)

    print(Delta1.shape)

    DW1 = alpha * np.dot(x.T, Delta1)

    synaptic_weights1 += DW1



    DW2 =  alpha * np.dot(y1.T, Delta)

    synaptic_weights2 += DW2







print('Y after training: ')

print(y1)


