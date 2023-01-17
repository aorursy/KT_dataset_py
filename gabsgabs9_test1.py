# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainpd = pd.read_csv("../input/train.csv")

# storing the Revenue column into a single variable

predictor_y = trainpd.revenue
predictor_y.describe()

#descreptive overview

plt.plot(predictor_y)

plt.show()
predictor_y.head(3)


over1b = []



for n in predictor_y:

    if n > 1000000000:

        over1b.append(n)

pd_y = pd.DataFrame(over1b)
pd_y.describe()
train_pd