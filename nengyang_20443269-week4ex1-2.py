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
mile = input("Enter miles:")  #input the number of mile
mile = float(mile)          #transform the mile format into float
km = float(mile/0.62137)    
meter = float(km*1000)     
if float(mile)>1:          # for 2 kinds of unit singular & plural possibility
    print(str(mile)+"miles is equivalent to")
    print('{:6.4f}km / {:6.1f}meters'.format(km,meter))
elif float(km*1000)>1:
    print(str(mile)+"mile is equivalent to")
    print('{:6.4f}km / {:6.1f}meters'.format(km,meter))
else:
    print(str(mile)+"mile is equivalent to")
    print('{:6.4f}km / {:6.1f}meter'.format(km,meter))
name = input("Please enter your name:")
age = input("Please enter your current age:")
age = int(age)+27   #age=(2047-2020)+current age
print( 'Hi {}! In 2047 you will be {}!'.format(name,age))