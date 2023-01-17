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
import numpy as np

import matplotlib.pyplot as plt

import random

import math

import seaborn as sns
n=30

rows, cols = (n, n) 

arr = np.zeros(shape=(n,n))



num_simulations = 200000

metal_stick_x = round(n/2)

metal_stick_y = round(n/2)

metal_thunder_cutoff = round(n ** (1. / 3))
avg=num_simulations/(n*n)

avg
for i in range (0,num_simulations): 

    x = random.randint(0,n-1)

    y = random.randint(0,n-1)

    #print ("x")

    #print (x)

    #print ("y")

    #print (y)

    

    if (abs(x-metal_stick_x) < metal_thunder_cutoff and abs(y-metal_stick_y) < metal_thunder_cutoff ):

        #print ("near stick")

        chance = random.randint(0,9)

        if (chance <4 ):

            x = metal_stick_x

            y = metal_stick_y

            #print ("hit stick")

        #else:

            #print ("hit near stick")

    else:

        if (abs(x-metal_stick_x) < (metal_thunder_cutoff+1) and abs(y-metal_stick_y) < (metal_thunder_cutoff+1) ):

            #print ("near stick")

            chance = random.randint(0,9)

            if (chance <2 ):

                x = metal_stick_x

                y = metal_stick_y

            #else:    

                #print ("hit stick")

        else:

            if (abs(x-metal_stick_x) < (metal_thunder_cutoff+2) and abs(y-metal_stick_y) < (metal_thunder_cutoff+2) ):

                #print ("near stick")

                chance = random.randint(0,9)

                if (chance <1 ):

                    x = metal_stick_x

                    y = metal_stick_y

                #else:    

                    #print ("hit stick")            

    

    arr[x][y] = arr[x][y]+1

    #print (arr[x][y])

plt.style.use('ggplot')

sns.heatmap(arr, cmap='viridis', vmin=(avg/(n*n)), vmax=(avg*(math.sqrt(n))))