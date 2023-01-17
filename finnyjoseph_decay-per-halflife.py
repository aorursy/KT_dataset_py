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
import matplotlib.pyplot as mb # for plotting graphs

import numpy as np #for creating array of values

import math as mh

print('enter the intial value')

Initial_value = int(input())



print('enter the number of half lives above')

NoOfHalfLives= int(input())

half_life = np.arange(0,NoOfHalfLives,0.1)#prints a range of half life

print(half_life)

x = half_life
y = Initial_value*((1/2)**(half_life))

print(y)
np.where(y<0)
mb.plot(x,y)

mb.xlabel("Half Lifes", fontdict=None, labelpad=None)

mb.ylabel("Quantities", fontdict=None, labelpad=None)

mb.title("Decay plot", fontdict=None, loc='center', pad=None)