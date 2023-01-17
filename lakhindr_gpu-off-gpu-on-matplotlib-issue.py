# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

l1 = ('Black', 'Hispanic', 'Asian', 'White')
l2 = (23.1, 20.7, 13.3, 12.9)    # sorted list..
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title("USA 2015: weighted poverty %")
plt.show()
