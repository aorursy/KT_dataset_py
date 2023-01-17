

import numpy as np

import matplotlib.pyplot as plot



 

x = np.arange(-10, 10, 0.5);

y = np.cos(x)



plot.plot(x,y);





import math

def f(x):

    return abs(math.cos(x))

     
#Test the function

f(-4.8)
lowerLimit = -5

upperLimit = 5



delta = 0.1

area = 0



while(lowerLimit<=upperLimit):

    area = area + (f(lowerLimit)*delta)

    lowerLimit = lowerLimit + delta

    

print(area)
