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



# 2. Write code to simulate the rolling of a dice for N time. N will be 1 to 1000. 

# The first simulation, you roll the dice once;  the second simulation, you roll the dice twice; ... the 1000th simulation, you roll the dice 1000 times.
np.random.choice



rolls = np.random.choice(a=[1,2,3,4,5,6], size=10, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

rolls.mean()
for N in range(1,1000):

    rolls = np.random.choice(a=[1,2,3,4,5,6], size=N, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

rolls
len(rolls)
# 3. Write code to calculate the mean of each simulation and append the means to a list variable.
mean = []

for N in range(1,1000):

    rolls = np.random.choice(a=[1,2,3,4,5,6], size=N, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

    mean.append(rolls.mean())

# 4. Make a plot that shows how the means fluctuate/converge as the number of rolls increase.
mean = []

for N in range(1,1000):

    rolls = np.random.choice(a=[1,2,3,4,5,6], size=N, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

    mean.append(rolls.mean())

fig, ax = plt.subplots(figsize=(12,8))

ax.set_ylim((1,6))

ax.plot(mean)
# 5. From the plot, what do you conclude?
