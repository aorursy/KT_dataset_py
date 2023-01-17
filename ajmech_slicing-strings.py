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
string1 = "Extract data from this sentence if you can..."



e0 = string1[0]

e1 = string1[1]

e2 = string1[0:5]

e3 = string1[8:8+4]

print(e0, e1, e2, e3)

print()

e4 = string1[::1]

print(e4)

e5 = string1[::2]

print(e5)

e6 = string1[::3]

print(e6)

print()



e7 = string1[::-1] # Reverses entire string



print("Reversed string = " + e7)



e8 = string1[44:44-3:-1] # Gets last three characters from string



print(e8)

print(e8 + string1) # Takes last three characters and puts them behind the string