# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

import time

def plt_dynamic(x, y,y_, ax,ticks,title, colors=['b']):

    ax.plot(x, y, 'b', label="Train Loss")

    ax.plot(x, y_, 'r', label="Test Loss")

    if len(x)==1:

        plt.legend()

        plt.title(title)

    plt.yticks(ticks)

    plt.xticks(ticks)

    fig.canvas.draw()

#     plt.pause(0.4)
import random

x = sorted(random.choices(range(100),k=100))

y = sorted(random.choices(range(100),k=100))

y_ = sorted(random.choices(range(100),k=100))

fig,ax = plt.subplots(1,1)

for i in range(90):

  plt_dynamic(x[:i+1],y[:i+1],y_[:i+1],ax,range(0,100,10),'one')

  # time.sleep(0.5)

plt.show()