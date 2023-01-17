# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = []

for _ in range(100):

    x = np.random.uniform(-10, 10)

    eps = np.random.normal(0, 0.1)

    y = 1.477 * x + 0.089 + eps

    data.append([x, y])

data = np.array(data)

print(data[:4,:])
x = data[:,0]

y = data[:,1]

m = data.shape[0]

w = np.sum(np.multiply(y, x - x.mean())) / (np.sum(x**2) - (np.sum(x)**2) / m)

b = (np.sum(y - w * x)) / m

print(w, b)
a = np.random.uniform(-10, 10)

print(a)

print(w * a + b)

print(a * 1.477 + 0.089)