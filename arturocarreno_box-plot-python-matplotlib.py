import matplotlib.pyplot as plt

import numpy as np
spread = np.random.rand(50) * 100

center = np.ones(25) * 50

flier_high = np.random.rand(10) * 100 + 100

flier_low = np.random.rand(10) * -100

data = np.concatenate((spread, center, flier_high, flier_low),0)
plt.boxplot(data)

plt.show()
plt.figure()

plt.boxplot(data,1)

plt.show()
plt.figure()

plt.boxplot(data, 0, 'gD')

plt.show()


plt.figure()

plt.boxplot(data, 0, '')

plt.show()
plt.figure()

plt.boxplot(data, 0, 'rs', 0)

plt.show()
plt.figure()

plt.boxplot(data, 0, 'rs', 0, 0.75)

plt.show()
spread = np.random.rand(50) * 100

center = np.ones(25) * 40

flier_high = np.random.rand(10) * 100 + 100

flier_low = np.random.rand(10) * -100

d2 = np.concatenate((spread, center, flier_high, flier_low), 0)

data.shape = (-1, 1)

d2.shape = (-1, 1)



data = [data, d2, d2[::2, 0]]



plt.figure()

plt.boxplot(data)



plt.show()