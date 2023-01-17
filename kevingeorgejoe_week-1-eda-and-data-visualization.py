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
data = pd.read_csv("../input/heart.csv")

data.head()
# Your code here
# Your code here
print(data.info())
# Your code here
# Your code here
# Your code here
# Your code here
df1, df2 = data[:100], data[100:]

print("df1.shape:", df1.shape)

print("df2.shape:", df2.shape)
# Your code here
# Your code here
import seaborn as sns
# Your code here
# Your code here