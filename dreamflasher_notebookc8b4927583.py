# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



x = np.matrix([[94,96,94,95,104,106,108,113,115,121,131]]).T

x_norm = x-np.mean(x)

y = np.matrix([[0.47, 0.75, 0.83, 0.98, 1.18, 1.29, 1.40, 1.60, 1.75, 1.9, 2.23]]).T

y_norm = y-np.mean(y)

w1 = np.linalg.inv(x_norm.T * x_norm) * x_norm.T * y_norm

print("w1=%s"%w1)

w0 = np.mean(y)-np.mean(x).T * w1

print("w0=%s"%w0)

sigma2=1/(len(x)-2) * np.sum(np.square(y-(w0+w1[0,0]*x)))

print("sigma=%s"%sigma2)

# Any results you write to the current directory are saved as output.