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
mushroom_data = pd.read_csv("../input/mushrooms.csv")
mushroom_data.head()
mushroom_data.info()
mushroom_data.describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
pd.plotting.parallel_coordinates(mushroom_data,"class")

edible = mushroom_data[mushroom_data["class"]=='e']
poisonous = mushroom_data[mushroom_data["class"]=='p']
edible.describe()
mushroom_class  = mushroom_data['class']
count = mushroom_class.value_counts()
count
count.plot.bar()
s={'p':'0','e':'1'}

# convert class category to numerical data
mushroom_data['class'] =mushroom_data['class'].map(s)
mushroom_data.head()





