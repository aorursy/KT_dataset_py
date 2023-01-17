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
miles = int(input('miles: '))

km = miles / 0.6217

meters = km * 1000

print(str(miles)+' miles equal to \n' + str('%.6f' %km)+ ' km / '+ str('%.1f' %meters)+' meters')
name = input('What is your name: ')

age = int(input('How old are you: '))

print('Hi '+ name + ' ! In 2047 you will be '+ str(2047-2020+age))