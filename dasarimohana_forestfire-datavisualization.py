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
# Import required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Reading the given data

df = pd.read_csv('../input/forest-fires-in-india/datafile.csv')

df.head()
# Checking null values in Dataset

df.isnull().mean()*100
# Fires occur in 2010-2011

df.plot(x = 'States/UTs',y= '2010-2011', kind = 'bar',figsize = (10,9), width = 0.5, color = 'blue')

plt.xlabel('States/UTs')

plt.ylabel('No.of Fires Occur')

plt.title('Occurance of fires in 2010-2011')
# Fires occur in 2009-2010

df.plot(x = 'States/UTs',y= '2009-10', kind = 'bar',figsize = (10,9), width = 0.5, color = 'green')

plt.xlabel('States/UTs')

plt.ylabel('No.of Fires Occur')

plt.title('Occurance of fires in 2009-2010')
# Fires occur in 2008-2009

df.plot(x = 'States/UTs',y= '2008-09', kind = 'bar',figsize = (10,9), width = 0.5, color = 'red')

plt.xlabel('States/UTs')

plt.ylabel('No.of Fires Occur')

plt.title('Occurance of fires in 2008-2009')
x = df['2010-2011'].mean()

print('Average fire occurance in 2010-2011 =',x)

y = df['2009-10'].mean()

print('Average fire occurance in 2009-2010 =',y)

z = df['2008-09'].mean()

print('Average fire occurance in 2008-2009 =',z)
# Pie Chart

labels = ['2010-2011','2009-10','2008-09']

size = [x,y,z]
plt.pie(size,labels=labels,autopct='%1.1f%%')

plt.title('Occurance of Fires over 3 years')

plt.show()
# Histogram

df.hist(rwidth=0.90,bins=20,figsize=(10,8))