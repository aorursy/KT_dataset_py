# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the pandas module

import pandas as pd



# Reading in the global temperature data

global_temp = pd.read_csv('../input/global_temperature.csv')
global_temp.head()
# Setting up inline plotting

%matplotlib inline



import matplotlib.pyplot as plt



# Plotting global temperature in degrees celsius by year

plt.plot(global_temp['year'], global_temp['degrees_celsius'])



# labels 

plt.xlabel('Year') 

plt.ylabel('Temperature') 

plt.title('Global Temperature in the last centuries ')

plt.show()