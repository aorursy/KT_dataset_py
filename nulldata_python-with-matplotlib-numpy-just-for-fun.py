import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings('ignore')

x = np.linspace(-2,2,1000)

y1 = np.sqrt(1-(abs(x)-1)**2)

y2 = -3 * np.sqrt(1-(abs(x)/2)**0.5)

plt.plot(x, y1, color="pink")

plt.plot(x, y2, color="pink")  

plt.xlim([-3.5, 3.5]);
x = np.linspace(-2,2,1000)

y3 = np.real(np.sqrt(abs(x)*(1-abs(x))))

y4 = np.real(-np.sqrt(1-np.sqrt(abs(x))))

plt.plot(x, y3, color="pink");

plt.plot(x, y4, color="pink");

plt.xlim([-2, 2]);
plt.plot(x, y3, color="blue")

plt.plot(x, y4, color="blue")  

plt.xlim([-2, 2]);