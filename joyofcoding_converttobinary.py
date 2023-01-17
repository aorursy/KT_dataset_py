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
#This code takes a whole number and converts it to 8-digit binary.
number = int(input("Input a whole number:"))
binary_number = str(int(number >= 128))+str(int(number % 128 >=64))+str(int(number % 128 % 64 >= 32))+str(int(number % 128 % 64 % 32 >= 16))+str(int(number % 128 % 64 % 32 % 16 >= 8))+str(int(number % 128 % 64 % 32 % 16 % 8 >= 4))+str(int(number % 128 % 64 % 32 % 16 % 8 % 4 >= 2))+str(int(number % 128 % 64 % 32 % 16 % 8 % 4 % 2 >= 1))
print(binary_number)
