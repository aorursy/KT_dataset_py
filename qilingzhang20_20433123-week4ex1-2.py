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
miles = float(input("Please enter the number in miles: "))
km = miles / 0.62137
m = 1000* km
print("%d miles is equivalent to \n%.4f km / %.1f meters." % (miles, km, m))
name = input("Please tell me your name: ")
age = int(input("Please tell me your age: "))
future_age = age + 27
print("Hi " + name + "! In 2047 you will be %s!" % (future_age))