# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random

import matplotlib.pyplot as plt



#print(*map(mean, zip(*a)))
def roll_dice(n_times, dice_rank):

    

    results = [  # Generate n_dice numbers between [1, dice_rank]

        random.randint(1, dice_rank)

        for n

        in range(n_times)

    ]

    #return results

    #print(results)

    return (sum(results)/len(results))
def doALot(num):

    results=[

        

        roll_dice(i,6)

        for i 

        in range(1,num)

    

    ]

    return results

    
a=doALot(1000)
fig, ax = plt.subplots(figsize=(12,8))

ax.set_ylim((1,6))

ax.plot(a)