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
from psutil import virtual_memory



import psutil



mem = virtual_memory()

print(mem.total)





print(psutil.cpu_percent())

print(psutil.virtual_memory())  # physical memory usage

print('memory % used:', psutil.virtual_memory()[2])
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv") 

# Preview the first 5 lines of the loaded data 

classes = [0,1]

data.head()

for clabel in classes:

    # Subset to the airline

    subset = data[data['open_channels'] == clabel]

    print(subset)

    # Draw the density plot

    sns.distplot(subset['signal'], hist = False, kde = True,

                 kde_kws = {'linewidth': 3},

                 label = clabel)

    

# Plot formatting

plt.legend(prop={'size': 16}, title = 'classes')

plt.title('Density Plot with classes')

plt.xlabel('signal (min)')

plt.ylabel('Density')

# Use the 'hue' argument to provide a factor variable

sns.lmplot( x="sepal_length", y="sepal_width", data=data, fit_reg=False, hue='open_channels', legend=False)

 

# Move the legend to an empty part of the plot

plt.legend(loc='lower right')