# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
temps = pd.read_csv("../input/melbtempsolympicpark/Temps.csv")
print(temps.head())
print(temps.shape)
print(temps.describe())

    
# get all years in temps
Years = temps['Year']
Years = sorted(set(Years))
print(Years)
# get all months in temps
Months = temps['Month']
Months = sorted(set(Months))
print(Months)
# get the mean temp
temp_mean = temps['Mean maximum temperature'].mean()
print(temp_mean)
temp_median = temps['Mean maximum temperature'].median()
print(temp_median) 
    
    
temps = temps.pivot(index='Year', columns='Month', values='Mean maximum temperature')
print(temps)