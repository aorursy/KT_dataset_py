import numpy as np # linear algebra

import pandas as pd 
W = np.array([1,-1,0,0.5]).transpose()

Xi = [np.array([1,-2,0,-1]).transpose(),np.array([0,1.5,0.5,-1]).transpose(), np.array([-1,1,0.5,-1]).transpose()]

d = [-1,-1,1]

c = 1

Error = 1

iteration = 0

i = 0

j = 0
while(Error != -0.0):

        net = sum(W.transpose()*Xi[i])

        o = (2/(1 + np.exp(-1*net)))-1

        o_ = ( 0.5 )*(1- (o**2) )

        error = d[i] - o

        print(round(error,1))

        Error = round(error,1)

        dw = c * error * o_ * Xi[i]

        W = W + dw

        iteration += 1

        i+=1

        if i > 2:

            i = 0

            j += 1

        if Error == -0.0:

            break
print("Final Weight Matrix : {}".format(W))

print("Iteration : {}".format(iteration))