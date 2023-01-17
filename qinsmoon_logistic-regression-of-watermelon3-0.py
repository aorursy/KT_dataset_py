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
data = np.array([[1, 0, 0, 0, 0, 0, .697, .46, 1], 

              [0, 0, 1, 0, 0, 0, .774, .376, 1], 

              [0, 0, 0, 0, 0, 0, .634, .264, 1], 

              [1, 0, 1, 0, 0, 0, .608, .318, 1], 

              [2, 0, 0, 0, 0, 0, .556, .215, 1], 

              [1, 1, 0, 0, 1, 1, .403, .237, 1], 

              [0, 1, 0, 1, 1, 1, .481, .149, 1], 

              [0, 1, 0, 0, 1, 0, .437, .211, 1], 

              [0, 1, 1, 1, 1, 0, .666, .091, 0], 

              [1, 2, 2, 0, 2, 1, .243, .267, 0], 

              [2, 2, 2, 2, 2, 0, .245, .057, 0], 

              [2, 0, 0, 2, 2, 1, .343, .099, 0], 

              [1, 1, 0, 1, 0, 0, .639, .161, 0], 

              [2, 1, 1, 1, 0, 0, .657, .198, 0], 

              [0, 1, 0, 0, 1, 1, .36, .37, 0], 

              [2, 0, 0, 2, 2, 0, .593, .042, 0], 

              [1, 0, 1, 1, 1, 0, .719, .103, 0]])

np.random.shuffle(data)

x = np.array(data[:, :-1])

y = np.array(data[:, -1])

print(x.shape, y.shape)
ratio = 0.6

x_train = np.array(x[: int(x.shape[0] * ratio), :])

y_train = np.array(y[: int(y.shape[0] * ratio)])

print(x_train.shape, y_train.shape)

x_test = np.array(x[int(x.shape[0] * ratio):, :])

y_test = np.array(y[int(y.shape[0] * ratio):])

print(x_test.shape, y_test.shape)
w_hat = np.zeros(x.shape[1] + 1)

x_train_hat = np.hstack((x_train, np.ones((x_train.shape[0], 1))))

n = 1

#batch_size = x_train_hat.shape[0] * 0.5

for _ in range(100):

    #梯度

    grad = 0

    for i in range(x_train_hat.shape[0]):

        e = np.exp(w_hat.T @ x_train_hat[i])

        grad += -(x_train_hat[i] * (y_train[i] - (e / (1 + e))))

    #更新梯度

    w_hat -= n * grad

print(w_hat)
x_test_hat = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

correct_count = 0

for i in range(x_test.shape[0]):

    y = 1 / (1 + np.exp(-(w_hat.T @ x_test_hat[i])))

    print(y, y_test[i])

    if ((y > 0.5 and y_test[i] == 1) or (y < 0.5 and y_test[i] == 0)):

       correct_count += 1

accuracy = correct_count / x_test.shape[0]

print(correct_count, accuracy)