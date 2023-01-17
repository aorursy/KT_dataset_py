# Parth Thacker
# Himanshu Joshi
# Pankita Jolapara

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.legend_handler import HandlerLine2D
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
temp=pd.read_excel("../input/temp.xls")

temp['JUN-SEP'].max()
y=[]
y.append(temp['ANNUAL'])
x=[]
x.append(temp['YEAR'])

import matplotlib.pyplot as plt


try_year=temp.values[111][:1]
try_temp=temp.values[111][1:13]
plt.ylabel("Tempreture")
plt.xlabel(try_year) 
x=[]
y=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
z=[]
n=[]
x=try_temp
for i in range(0,12):
    z.append(18)
for i in range(0,12):
    n.append(24)
plt.plot(y,n)
plt.plot(y,x)
plt.plot(y,z)
plt.scatter(y,x)
plt.scatter(y,x)
plt.show()


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

max1=[]
min1=[]
avg=[]
year=[]
anul=[]

for i in range(len(temp)):
    max1.append(temp.values[i][1:13].max())
for i in range(len(temp)):
    min1.append(temp.values[i][1:13].min())
for i in range(len(temp)):
    avg.append(temp.values[i][1:13].mean())
for i in range(len(temp)):
    year.append(temp.values[i][0])
for i in range(len(temp)):
    anul.append(temp.values[i][14])
    
plt.plot(year,max1,color='red')
plt.plot(year,avg,color='Green')
plt.plot(year,min1,color='blue')
plt.plot(year,anul,color='Orange')

red_patch1 = mpatches.Patch(color='red', label='Hot')
red_patch2 = mpatches.Patch(color='Orange', label='Anual data of dataset')
red_patch = mpatches.Patch(color='blue', label='Cold')
red_patch3 = mpatches.Patch(color='Green', label='Average of 12 months')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,handles=[red_patch,red_patch1,red_patch2,red_patch3])



plt.show() 
