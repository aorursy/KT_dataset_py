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
import matplotlib as mlp

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib2tikz as mattikz

rainDatainMM1 = [9.63, 9.93 , 9.23, 11.13,

 9.28, 9.14, 9.51, 9.93,

 10.61, 10.13, 10.72, 9.68]

months = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(months, rainDatainMM1, color='c', linewidth=3.0);

#mattikz.save("tikzCode/1Line.tikz")

rainDatainMM1 = [9.63, 9.93 , 9.23, 11.13,

 9.28, 9.14, 9.51, 9.93,

 10.61, 10.13, 10.72, 9.68]

months1 = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(months1, rainDatainMM1, color='g', linewidth=2.0)

plt.title("Monthly Rain in mm Plot")

plt.ylim(0,12)

plt.text(x=5.7,y=5.5,s='PK');

#mattikz.save("tikzCode/3Line.tikz")

#plt.show()
