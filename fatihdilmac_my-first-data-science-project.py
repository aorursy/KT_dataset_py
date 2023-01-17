# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read data from movies_metadata.csv file and store in data variable
data=pd.read_csv('../input/movies_metadata.csv')
#I want to know my data
data.info()
data.describe()
# I want to analyse the first 5 sample in given data
data.head()
count=0 # I will find number of  tr movies
for index,value in data[['original_language']][0:].iterrows():
   if value[0]=='tr':
    count=count+1
    print(index," -->",value[0])
   
print("----------------------------------------------------------------")
print("Numbers of TR movies",count)

data.corr()
#correlation map
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

