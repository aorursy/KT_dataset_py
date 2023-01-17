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
data_dir = "../input/"

df = pd.read_csv(data_dir + "train.csv", usecols=["ID_code", "target"])

df.reset_index(inplace=True)

df.rename(columns={"index": "order"}, inplace=True)

df.head(3)
x = df[df["target"] == 1]["order"] # All positive rows

x2 = np.diff(x) # distance between 2 row orders



import matplotlib.pyplot as plt

plt.hist(x2, bins=100);