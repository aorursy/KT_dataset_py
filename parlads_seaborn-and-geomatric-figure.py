# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline
people = np.arange(1, 11)
total_cost = np.full_like(people, 40.0)
ax= plt.gca()
ax.plot(people, total_cost)
ax.set_xlabel('# of people')
ax.set_ylabel('Cost')
total_cost = 80.0 * people + 40.0
pd.DataFrame({'total cost': total_cost} ,index = people)
ax= plt.gca()
ax.plot(people, total_cost)
ax.set_xlabel('# of people')
ax.set_ylabel('Cost')
print(np.__file__)
#from mlwpy_video_extras import high_school_style

%matplotlib inline
m,b = 1.5, -3

xs = np.linspace(-3, 3, 100)
ys = m * xs + b

ax = plt.gca()
ax.plot(xs, ys, 'y')

ax.plot(0 , -3, 'rp')
ax.plot(2, 0 ,'ro')

ys = 0 * xs + b
ax.set_ylim(-4, 4)