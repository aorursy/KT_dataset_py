# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

letters = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        letter = open(os.path.join(dirname, filename)).read()

        letters.append(letter)

print(len(letters))



# Any results you write to the current directory are saved as output.
print(letters[0])
import matplotlib.pyplot as plt



x = []

y = []



for i in range(len(letters)):

    x.append(i)

    y.append(len(letters[i]))

plt.xlabel('Letter Number')

plt.ylabel('Length of letter')

plt.plot(x,y)



plt.show()
