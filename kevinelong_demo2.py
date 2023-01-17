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
total = 0
total = total - 5
total -= 5
total
from random import randint
def roll(quantity: int = 3, sides: int = 6) -> int:
    total = 0
    # TODO Put your code here.
    
    # loop to quantity - using for in range ...
    for i in range(0, quantity):
        
        # generate random integer from 1 to sides
        # add to the total
        total = total + randint(1,sides)
        
    return total

# Test
print( roll(quantity=1,sides=30) )

# def track(times: int, quantity: int = 3, sides: int = 6) -> {int:int}:
def track(times, quantity = 3, sides = 6):

    #Create empty dictionary
    history = {}
#     history = dict()
    
    # TODO Put your code here
    
    for i in range(0,times):

        v = roll(quantity, sides)
        
        if v not in history:
            # set to 1 as its the first one seen
            history[v] = 1
        else:
            #weve seen it before so we increment
            # history[v] = history[v] + 1
            history[v] += 1

    return history


# TEST
print(track(1000))

SCREEN_SIZE = 80

def chart(times=1000, quantity=3, sides=6):
    
    data = track(times, quantity, sides)

    values = data.values()
    print(values)

    m = max(values)
    print(m)
    
    ratio = m // SCREEN_SIZE
    print(f"ratio is {ratio}")
    
    keys = data.keys()
    keys = sorted(keys)
    
    for k in keys:    
        if k in data:
            d = data[k]
            scaled = d // ratio
            print("#" * scaled, k, f"({d})" )
        else:
            print(i)


chart(10000,3,18)
