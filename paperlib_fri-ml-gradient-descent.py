%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy  as np # -- linear algebra

import pandas as pd # -- data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt
data = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [2.1, 3.9, 5.7, 8, 9.9, 12.1, 14.2, 15.9, 18.1]]
data[1][4]
inputs  = data[0]

outputs = data[1]
# -- this is the "gradient" of the cost function at any given weigth (w)

# -- -> this function returns the value of the derivative of the cost

# -- function at w

def gradient(w):

    result = 0

    

    for (input, output) in zip(inputs, outputs):

        result += w * input - output

        

    result /= len(inputs)

    

    return result
weight = -20

learning_rate = 0.1



samples = np.arange(-10, 10)

plt.axis([-10, 10, -10, 10])



fig = plt.figure()

camera = Camera(fig)



while abs(gradient(weight)) > 0.0000001:

    weight -= learning_rate * gradient(weight)

    

    print(weight)

    plt.plot(samples, weight * samples)

    camera.snap()

    

    #time.sleep(2)



animation = camera.animate()

animation.save('celluloid_minimal.gif', writer = 'imagemagick')

#def gradient

#weight = 10

#

#derivative(weight)
animation = camera.animate()

animation
from IPython.display import Image, display

display(Image(url='celluloid_minimal.gif'))
#np.random.uniform(-0.5, 0.5)

outputs = None

inputs = np.arange(1, 101)

#outputs = (inputs * 2) + np.random.uniform(-0.5, 0.5)

#outputs

#(2*inputs) + np.random.randn(10)

#(inputs * 2) + np.random.randn(10)

#np.random.randn(10)

outputs = (inputs * 2) + np.random.uniform(-1.0, 1.0, 100)

#outputs

for _ in range(4):

    outputs = np.vstack((outputs, (inputs * 2) + np.random.uniform(-1.0, 1.0, 100)))

    

outputs

    #print(outputs)

#outputs

#outputs = (inputs * 2) + np.random.uniform(-0.3, 0.3, 10)

#outputs

#other = (inputs * 2) + np.random.uniform(-0.3, 0.3, 10)

#np.append([outputs], [other], axis=0)



for sample in outputs:

    plt.plot(inputs, sample, 'ro')



outputs[1]
np.random.uniform(-1.0, 1.0, 10)