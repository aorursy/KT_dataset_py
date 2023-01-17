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
import matplotlib.pyplot as plt
data = [[2,81], [4,93], [6,91], [8,97]]

x = [i[0] for i in data]

y = [i[1] for i in data]
plt.figure(figsize=(8,5))

plt.scatter(x,y)

plt.show()
x_data = np.array(x)

y_data = np.array(y)
a = 0

b = 0
lr = 0.05
epochs = 2001
for i in range(epochs):

    y_pred = a * x_data + b

    error = y_data - y_pred

    a_diff = -(1/len(x_data)) * sum(x_data * (error))

    b_diff = -(1/len(x_data)) * sum(y_data - y_pred)

    

    a= a-lr*a_diff

    b= b-lr*b_diff

    

    if i % 100 == 0:

        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i,a,b))

        
y_pred = a * x_data +b

plt.scatter(x, y)

plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])

plt.show()