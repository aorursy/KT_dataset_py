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
import matplotlib.pyplot as plt
fgpOfLonzo = [0.286,0.375,0.462,0.4,0.571,0.364,0.6,0.25,0.333,0.333,0.308,0.455,0.167,0.667,0.111,0.6,0.222,0.333,0.429,0.308,0.4,0.545,0.333,0.308,0.364,0.5,0.625,0.474,0.571,0.364,0.429,0.333,0.304]
print('Number of games:' + str(len(fgpOfLonzo)))

print('Mean of field goal percentage during this season:' + str(np.mean(fgpOfLonzo)))

print('Median of field goal percentage during this season:' + str(np.median(fgpOfLonzo)))

from statistics import mode 

print("mode: ", mode(fgpOfLonzo))
plt.hist(fgpOfLonzo,6) # Even though the homework says 6 bins, I've put 10 to see the histogram better

plt.xlabel("FG% in a Game")

plt.ylabel("# of games")

plt.title("Lonzo Ball FG% Distribution")

plt.show()