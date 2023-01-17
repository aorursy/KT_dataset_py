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
print("Hello World")
import matplotlib.pyplot as plt
# Diamond Joe Flacco 

touchdowns = [14, 21, 25, 20, 22, 19, 27, 14, 20, 18, 12]

print("number of seasons: ", len(touchdowns)) #len() is the length of the list
print("Joe Flacco is Elite!")
print("range: ", np.min (touchdowns), "-", np.max(touchdowns))
print("touchdowns per seasons: ", np.mean(touchdowns))

print("median: ", np.median (touchdowns))
from statistics import mode #import the function 'mode'

print ("mode: ", mode(touchdowns))
print("mode: 20.0")
plt.hist(touchdowns, 6)

plt.xlabel ("Touchdowns")

plt.ylabel("N")

plt.title("Joe 'Elite' Flacco's Touchdown Distribution")

plt.show()