import numpy as np

import matplotlib.pyplot as plt



xs = np.linspace(0.00000001, 1, 1000)

ys = np.log(xs)



plt.plot(xs, ys, label='Actual Log')

plt.plot(xs, -ys, label='Neg. Log')



plt.xlabel('Probabilities [0+e, 1]')

plt.legend()

plt.show()