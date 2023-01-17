# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def AND(x1, x2):

    w1, w2, theta = 0.5, 0.5, 0.7

    tmp = x1*w1 + x2*w2

    if tmp <= theta:

        return 0

    elif tmp > theta:

        return 1
print(AND(0, 0))

print(AND(1, 0))

print(AND(0, 1))

print(AND(1, 1))
import numpy as np

x = np.array([0, 1])

w = np.array([0.5, 0.5])

b = -0.7

print(x*w)

print(np.sum(w*x))

print(np.sum(w*x)+b)
def AND(x1, x2):

    x = np.array([x1, x2])

    w = np.array([0.5, 0.5])

    b = -0.7

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1
def NAND(x1, x2):

    x = np.array([x1, x2])

    w = np.array([-0.5, -0.5])

    b = 0.7

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1
def OR(x1, x2):

    x = np.array([x1, x2])

    w = np.array([0.5, 0.5])

    b = -0.2

    tmp = np.sum(w*x) + b

    if tmp <= 0:

        return 0

    else:

        return 1
def XOR(x1, x2):

    s1 = NAND(x1, x2)

    s2 = OR(x1, x2)

    y = AND(s1, s2)

    return y
print(XOR(0, 0))

print(XOR(1, 0))

print(XOR(0, 1))

print(XOR(1, 1))