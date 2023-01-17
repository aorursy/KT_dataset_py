import numpy as np 

import matplotlib.pyplot as plt
def sigmoid(x):    

    return (1/(1+np.exp(-x)))

def softmax(x):

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0) 
array2 = np.arange(-10,11,1)

print(array2)
sigmoid(array2)
plt.figure(figsize=(8,6))

plt.plot(array2, sigmoid(array2), 'r')
print(sigmoid(0))
print(softmax(array2))
plt.plot(array2, softmax(array2), 'g')
plt.plot(array2, sigmoid(array2), 'r', label='sigmoid')

plt.plot(array2, softmax(array2), 'g', label='softmax')

plt.legend()
sigmoid(0)
sigmoid(5)
sigmoid(-5)
softmax(0)
softmax(5)
softmax([0,0])
softmax([1,1,1])
softmax([1,1,2])
softmax([1,1,5])
print(softmax([1,1,2,5,10]))

print("sum : ", sum(softmax([1,1,2,5,10])))
scores2D = np.array([[1, 2, 3, 6],

                     [2, 4, 5, 6],

                     [3, 8, 7, 6]])
print(sigmoid(scores2D))
print(sum(sigmoid(scores2D)))
print(softmax(scores2D))
print(sum(softmax(scores2D)))