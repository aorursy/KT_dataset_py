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
test=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')
#Show first 20 rows of the file
test.head(20)
#Calculate the median of column Slope
mean = test['Slope'].mean()
#Calculate the STD of column Slope
std = test['Slope'].std()

print("The mean of the column = %0.2f , The STD of the column = %0.2f" % (mean , std))
test.info()
# Import matplotlib to show the graph
import matplotlib.pyplot as plt

#Plot the frequency of values of column Elevation 
test['Elevation'].hist(bins=100)
plt.show()
#Checking if some column has null value
test.isna().sum()
#Describe each column
for col in test.columns:
    print(test[col].describe(), "\n")