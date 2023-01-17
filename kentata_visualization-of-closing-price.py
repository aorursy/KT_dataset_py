# In this notebook, I show how to visualize the closing price



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")
print("train size is: {}".format(train.shape))

train.head()
train = train.iloc[::-1]

train.head()
train = train.set_index('Date')

train.head()
train.describe()
ax = train['Close'].plot(style=['-'])

ax.lines[0].set_alpha(0.8)

# y-axis starts from zero. Otherwise, graph will be misleading

ax.set_ylim(0, np.max(train['Close'] + 100))

plt.xticks(rotation=90)

plt.title("linear scale")

ax.legend()
ax = train['Close'].plot(style=['-'])

ax.lines[0].set_alpha(0.3)

ax.set_yscale('log')

ax.set_ylim(0, np.max(train['Close'] + 100))

plt.xticks(rotation=90)

plt.title("logarithmic scale")

ax.legend()
