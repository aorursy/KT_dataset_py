# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def my_addition(a, b=0):
    print(a + b)
my_addition(10)
my_addition(10, 20)
my_addition(10, 30)
def check_value_within_condition(a):
    if ((a >= 10) & (a <= 20)):
        print(a, "is between 10 and 20")
    elif ((a >= 20) & (a <= 30)):
        print(a, "is between 20 and 30")
    elif ((a >= 30) & (a <= 40)):
        print(a, "is between 30 and 40")
    elif ((a >= 40) & (a <= 50)):
        print(a, "is between 40 and 50")
    else:
        print(a, "is not within condition")
check_value_within_condition(25)
check_value_within_condition(45)
check_value_within_condition(35)
check_value_within_condition(15)
check_value_within_condition(5)
check_value_within_condition(55)
check_value_within_condition(10)
check_value_within_condition(40)
