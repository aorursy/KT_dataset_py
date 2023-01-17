# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/parking-citations.csv",low_memory=False)
# print(data.head())
# Any results you write to the current directory are saved as output.
print(data.shape)
print(data.dtypes)
import matplotlib.pyplot as plt
data.plot(kind='scatter',x='Latitude',y='Longitude',color='red')
plt.show()
import matplotlib.pyplot as plt
data.plot(x='Ticket number',y='Fine amount',style='o')
plt.show()