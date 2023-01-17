import numpy as np

import matplotlib.pyplot as plt
y = np.arange(-3.14, 3.14, 0.1)

x = np.sin(y)

plt.plot(y,x)

plt.title('Sine Wave')

plt.xlabel('time')

plt.ylabel('sin function')

plt.show()