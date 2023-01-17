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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib

import matplotlib.pyplot as plt

#Data for plotting



er = 2.55

n1 = 377

n2 = 236



theta_i = np.arange(0.0,np.pi/2,0.01)       #rad

theta_t = np.arcsin(np.sqrt(1/er)*np.sin(theta_i))      #rad



fig, ax = plt.subplots()

#Perpendicular

reflect_coeff_1 = ((n2*np.cos(theta_i))-(n1*np.cos(theta_t))) / ((n2*np.cos(theta_i))+(n1*np.cos(theta_t)))

ax.plot(theta_i*(180/np.pi), np.abs(reflect_coeff_1),label='Perpendicular')

plt.xticks(np.arange(0, 91, step=10))



#Parallel

reflect_coeff_2 = ((n2*np.cos(theta_t))-(n1*np.cos(theta_i))) / ((n2*np.cos(theta_t))+(n1*np.cos(theta_i)))

ax.plot(theta_i*(180/np.pi), np.abs(reflect_coeff_2),label='Parallel')



#Set

plt.legend()

ax.set(xlabel='Incident angle', ylabel='Reflection coeff',

       title='Reflection Coefficient Magnitude')

ax.grid()

plt.show()
