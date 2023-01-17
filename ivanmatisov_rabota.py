import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
np.random.random()
a = np.random.rand(8,4)
print(a)
a[:,-1].shape
a = 2.4

b = 3.7

c = -1.8
x = np.arange(-10,10+0.5,0.5)
y = a * x**2 + b*x + c

print(f"x={x}, y={y}")
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(x,y,"r-", linewidth=5.0);

plt.grid()

plt.xlabel("X");

plt.ylabel("Y")

plt.title("Заголовок");