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
import numpy as np

import math

from random import randint



#Creating the initial population.



new_population = np.random.randint(2, size=(4, 6))



print(new_population)
chromosome = new_population[:1]

print(chromosome)



#sign_bit = new_population[2][0]



#print(len(new_population))



fit_v = [0] * 4



for i in range(4):

        sign_bit = chromosome[i:,0]

        fit_v[i] = chromosome[i:,1]*pow(2,4) + chromosome[i:,2]*pow(2,3) + chromosome[i:,3]*pow(2,2) + chromosome[i:,4]*pow(2,1) + chromosome[i:,5]*pow(2,0)

        value = fit_v[i]

        if (sign_bit == 1):

            value = -value

        else:

            value = value

        i=i+1



print(sign_bit)            

print(fit_v[0])

print(fit_v[1])

print(fit_v[2])

print(fit_v[3])