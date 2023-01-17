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
price=pd.read_csv("../input/prices.csv",index_col=0)
price=price.drop(['symbol'], axis=1)
price.head()
# Read in the data
#data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
price.index = pd.to_datetime(price.index)
print(price.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in price:
    price[column].plot(ax=ax, label=column)
ax.legend()
plt.show()


