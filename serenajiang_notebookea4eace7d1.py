# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

print("done")
df.sample(frac=.001).plot(kind='hexbin', x='x', y='y', gridsize=25)
df.sample(frac=.001)["time"].plot(kind="density")
places = df.groupby('place_id')

dists = places.max() - places.min()
stds = places.std()

# IDK why it won't let me plot this...
plt.violinplot([dists["x"], dists["y"]])
places.min().mean()
# missing values

df.isnull().sum()
# max values

df.max()
# min values

df.min()