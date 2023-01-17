# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
x = np.array([1, 0.5, 0.2]) # 入力

y = np.array([0, 0, 1]) # 正解

w = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 1]]) # パラメーター
def softmax(x):

    t= np.exp(x)

    return t / t.sum()
t = softmax(w.dot(x)) # 確率の実装
def cross_entropy(y, t):

    return - y.dot(np.log(t))

    
cross_entropy(y, t)