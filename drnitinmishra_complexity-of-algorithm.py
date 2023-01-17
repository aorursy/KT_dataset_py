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
import time 

import matplotlib.pyplot as plt 
def a1(n):
    start_time = time.time()
    for i in range(1,n):
        print(n*n)
    end_time = time.time()   
    return end_time-start_time

    
def a2(n):
    start_time = time.time()
    for i in range(1,n):
        for j in range(1,i):
            print(n)
    end_time = time.time()   
    return end_time-start_time

    
x=[]
y=[]
for n in range(1,100):
    x.append(n)
    y.append(a2(n))
    
   
plt.xlabel('List Length') 
plt.ylabel('Time Complexity') 
plt.plot(x, y, label ='Algorithm') 
plt.grid() 
plt.legend() 
plt.show() 
plt.xlabel('List Length') 
plt.ylabel('Time Complexity') 
plt.plot(x, y, label ='Algorithm') 
plt.grid() 
plt.legend() 
plt.show() 