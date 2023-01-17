# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
z = np.array([1,2,3,4])
fz = z*2 +1
plt.plot(z,fz)
plt.grid()
def sigmoid(z):
    return 1/(1+np.exp(-z))

z = np.linspace(-10,10,100)
fz = sigmoid(z)
plt.plot(z,fz)
plt.grid()
def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z)+np.exp(-z))

z = np.linspace(-10,10,100)
fz = tanh(z)
plt.plot(z,fz)
plt.grid()
def ReLU(z):
    z_zero = np.array([])
    z_max = np.array([])
    z_result = np.array([])
    for i in z:
        if i > 0:
            z_max = np.append(z_max,i)
        else:
            i = 0
            z_zero = np.append(z_zero,i)
    z_result = np.concatenate((z_zero,z_max))
    return z_result

z = np.linspace(-1,1,100)
fz = ReLU(z)
plt.plot(z,fz)
plt.grid()
def leakyReLU(z,alpha = 0.05):
    z_zero = np.array([])
    z_max = np.array([])
    z_result = np.array([])
    for i in z:
        if i > 0:
            z_max = np.append(z_max,i)
        else:
            i = alpha * i 
            z_zero = np.append(z_zero,i)
    z_result = np.concatenate((z_zero,z_max))
    return z_result

z = np.linspace(-1,1,100)
fz = leakyReLU(z)
plt.plot(z,fz)
plt.grid()
