import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline

import numpy as np

x = np.linspace(0, 100, 100)

m = np.linspace(100, 0, 100)

with np.errstate(divide='ignore'): # ignore division by zero

    y = x / m





plt.plot(x, y, 'r') # 'r' is the color red

plt.xlabel('% Resource Busy')

plt.ylabel('Wait Time')

plt.title('Wait Time = (% Busy)/(% Idle)')

plt.show()


