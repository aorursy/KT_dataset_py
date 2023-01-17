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
print((np.sqrt(2)**2)-2);
#### If tolerance is selected lower than 10^-16

tolerance = 10**-15;

if (np.sqrt(2))**2 - 2 < tolerance:

    result = 0;

else:

    result = (np.sqrt(2))**2 - 2;

print(result)
# If precision is selected lower than 10^-16

# using "%" to print value till 2 decimal places  

print ("The value of number till 2 decimal place(using %) is : ",end="") 

print ('%.2f'%((np.sqrt(2))**2 - 2)) 

  

# using format() to print value till 2 decimal places  

print ("The value of number till 2 decimal place(using format()) is : ",end="") 

print ("{0:.2f}".format((np.sqrt(2))**2 - 2)) 

  

# using round() to print value till 2 decimal places  

print ("The value of number till 2 decimal place(using round()) is : ",end="") 

print (round(((np.sqrt(2))**2 - 2),2))