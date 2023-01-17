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
#1D array

import numpy

a = numpy. array([3,5,7,9,11])

print(a) 
#2D array

import numpy as np

a = np.array([[7,8,9],[4,7,8]]) 

print(a) 
#Boolean array

import numpy as np

a = np.array([2,6,3.4,0, True, False, ], dtype = bool) 

print(a) 
#Extracting odd numbers

import numpy as np

a = np. array([[2,4,7,9], [1,6,8,5]]) 

b=a[a%2==1]

print(b) 
#common elements

import numpy as np

a = np.array([2,4,6,8,9])

b = np.array([1,2,5,6,9]) 

print("Common values between two arrays:")

print(np.intersect1d(a,b))
