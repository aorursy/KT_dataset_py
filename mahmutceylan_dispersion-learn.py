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
salary = [100,13,44,23,56,13,68]
#Range

Max = np.max(salary)

Min = np.min(salary)

Range = Max-Min

print("Range=",Range)
#Variance

var = sum((salary - np.mean(salary))**2)/len(salary)

var = round(var,2)

print("Variance= {}".format(var))

 #Standart Deviation

std = np.sqrt(sum((salary - np.mean(salary))**2)/len(salary))

std = round(std,2)

print("Std= {}".format(std))