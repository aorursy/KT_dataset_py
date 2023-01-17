# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read in data
df = pd.read_csv('../input/HardcoreDelveCurrency_08.csv')
df.info()
df.head(5)
df.describe(include = 'all')
# dataframe with only chromatic orbs
chrom_value = df[(df.Get == 'Chromatic Orb') & (df.Confidence == 'High')]

# get date data into something plotable (aug,sep,oct,nov)
count_row = chrom_value.shape[0]
# 94 rows of data for chroms
# q4 = 94, q3 = 71, q2 = 47, q1 = 24

# plot chromatic orb purchased price against date (x axis) and value in chaos (y)
plt.xlabel('Time')
plt.ylabel('Value in Chaos')
plt.title('Chromatic Orb value during Delve HC in Chaos')
plt.xticks([24, 47, 71, 94], ["Sep", "Oct", "Nov", "Dec"])
plt.plot(chrom_value.Date, chrom_value.Value)
# what currency is purchased with Chaos
get_currency_all = df.Get.unique()

# graph all get currency that shows value overtime
for i in get_currency_all:
    currency = df[(df.Get == i) & (df.Confidence == 'High')]
    a = plt.plot(currency.Date, currency.Value)
    count_row = currency.shape[0]
    #q1
    q1 = (count_row / 2) / 2
    #q2
    q2 = (count_row / 2)
    #q3
    q3 = (count_row / 2) + q1
    #q4
    q4 = count_row
    plt.xticks([q1, q2, q3, q4], ["Sep", "Oct", "Nov", "Dec"])
    plt.title(i)
    plt.show()
    
