from IPython.display import Image

import os

!ls ../input/
Image('../input/pictures1/snn.png')
#Import numpy for numerical calculations



import numpy as np
input_weights = np.around(np.random.uniform(-5,5,size=6), decimals=2)

bias_weights = np.around(np.random.uniform(size=3), decimals=2)
print(input_weights)

print(bias_weights)
x_1 = 0.5 #input 1

x_2 = 0.82 #input 2



print('Input x1 is {} and Input x2 is {}'.format(x_1,x_2))
z_11 = x_1 * input_weights[0] + x_2 * input_weights[1] + bias_weights[0]



print('The linear combination of inputs at the first node of the hidden layer is {}'.format(z_11))
z_12 = x_1 * input_weights[2] + x_2 * input_weights[3] + bias_weights[1]



print('The linear combination of inputs at the second node of the hidden layer is {}'.format(z_12))
Image('../input/pictures1/relu.png')
a_11 = max(0.0, z_11)



print('The output of the activation function at the first node of the hidden layer is {}'.format(np.around(a_11, decimals=4)))
a_12 = max(0.0, z_12)



print('The output of the activation function at the second node of the hidden layer is {}'.format(np.around(a_12, decimals=4)))
z_2 = a_11 * input_weights[4] + a_12 * input_weights[5] + bias_weights[2]



print('The linear combination of inputs at the output layer is {}'.format(z_2))
Image('../input/pictures1/sigmoid.png')
y = 1.0 / (1.0 + np.exp(-z_2))



print('The output of the network for the given inputs is {}'.format(np.around(y, decimals=6)))