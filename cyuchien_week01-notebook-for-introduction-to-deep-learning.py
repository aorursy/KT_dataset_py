# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
%matplotlib inline
def QuadraticKernel(X):
    print('Shape X:', X.shape)
    result = np.array([[r[0]**2, r[1]**2, r[0]*r[1], r[0], r[1], 1] for r in X])
    print('Shape X\':', result.shape)
    return result
x = np.array([[1, 2],
              [3, 4]])
print(QuadraticKernel(x))
def classification_acc(wx, y):
    result = (wx == y)
    return np.sum(result) / wx.shape[0]
classification_acc(np.array([1, 2, 3, 4]), np.array([1, 3, 3, 4]))
x = np.linspace(-6, 6, 100)
y_step = np.where(x >=0, 1, -1)
y_sigmoid = 1 / (1+ np.exp(-x))

plt.figure(figsize=(10, 2))
plt.subplot(121)
plt.grid()
plt.xlabel('step function')
plt.plot(x, y_step)
plt.subplot(122)
plt.grid()
plt.plot(x, y_sigmoid)
plt.xlabel('sigmoid function')
def softmax(ori):
    exp_ori = np.exp(ori)
    sum_exp = np.sum(exp_ori)
    return exp_ori / sum_exp
softmax(np.array([1.2, 9.3, -2.4, -5.6]))
def cross_entropy_of_two_classes(y, prob):
    num_sample = y.shape[0]
    err = (-1/num_sample) * np.sum(y * np.log(prob) + (1-y) * np.log(1-prob))
    return err

y = np.array([1, 0, 0])
good_y = np.array([0.9, 0.1, 0.05])
bad_y = np.array([0.1, 0.9, 0.95])

# small error
print('answer', y,'prob', good_y, 'error', cross_entropy_of_two_classes(y, good_y))
# large error
print('answer', y, 'prob', bad_y, 'error', cross_entropy_of_two_classes(y, bad_y))
from IPython.display import Image
Image("../input/optimizer_compare_naive.png", width=800)
Image("../input/Regularizer.png", width=400)
