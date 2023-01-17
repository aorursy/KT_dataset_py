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
from sklearn import svm

svc = svm.SVC(gamma=0.001, C=100.)
from sklearn import datasets

digits = datasets.load_digits()

print(digits.DESCR)

digits.images[0]
import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
digits.target.size

plt.subplot(321)

plt.imshow(digits.images[1791], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(322)

plt.imshow(digits.images[1792], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(323)

plt.imshow(digits.images[1793], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(324)

plt.imshow(digits.images[1794], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(325)

plt.imshow(digits.images[1795], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(326)

plt.imshow(digits.images[1796], cmap=plt.cm.gray_r,

interpolation='nearest')
svc.fit(digits.data[1:1790], digits.target[1:1790])



svc.predict(digits.data[1791:1796])
digits.target[1791:1796]
