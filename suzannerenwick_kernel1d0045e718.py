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
import numpy as np

import matplotlib.pyplot as plt
imported_data = np.recfromtxt('/kaggle/input/cd-data/cd_data.txt', skip_header=1)



data = list(imported_data)



print(data)
for value in data:

    print(value)
x_values = []

y_values = []



for value in data:

    x = value[0]

    y = value[1]

    

    x_values.append(x)

    y_values.append(y)

    

print(x_values)

print()

print(y_values)
plt.plot(x_values,y_values)
plt.plot(x_values, y_values, marker='^', color='purple', markerfacecolor='pink', markeredgecolor='purple', linewidth=2)
plt.plot(x_values, y_values, marker='^', color='purple', markerfacecolor='pink', markeredgecolor='purple', linewidth=2)

plt.xlabel('x-axis (Units)', labelpad=10)

plt.ylabel('y-axis (Units)')



plt.savefig('cd_data.png', dpi=600, transparent=True)
