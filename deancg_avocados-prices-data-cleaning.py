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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from pylab import rcParams

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

rcParams['figure.figsize'] = 12, 8

#This is just some basic imports of the libraries we will be using.



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = None # We could specify "none" inplace of 1000 but for this excersise we can stick with 1000 for learning purposes

df1 = pd.read_csv('../input/avocado-prices-2020/avocado-updated-2020.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'avocado-updated-2020.csv'

nRow, nCol = df1.shape

print('There are {nRows} rows and {nCol} columns') #WHY NOT WORK??? 
df1.head(5) #THIS GIVES US THE VERY TOP 5 ROWS FROM OUR DATA SET
df1.describe() #This function is great for pulling out count,mean,min & max
df1.drop(['4046', '4225', '4770'], axis=1).head() #This is a useful function for removing any irrelivent data from the df
pd.value_counts(df1['type']).plot.bar()
df1['average_price'].mean() #THIS FUNCTION GIVES US THE MEAN() OF WHAT EVER PARAMETER WE PUT INTO []
df1.shape
df1.size
df1.year.unique() # DISPLAYS THE UNIQUE YEARS IN THE DATASET
df1.geography.unique() # DISPLAYS ALL THE REGIONS IN THE DATASET
df1.geography.value_counts() # THIS THEN GIVES US THE COUNT FOR EACH REGION
g = df1[["year", "average_price"]].groupby("year").mean()



g.plot();
df1['average_price'].hist()
df1.plot(x='average_price', y='total_volume', kind='scatter')
df1.isnull().sum() # A good function to use to see if any data is missing
df1.dropna() # THIS IS THE FUNCTION USED IF WE DID HAVE ANY MISSING DATA - BUT WE DON'T WITH THIS DATASET

#df1.fillna(x) # AND THIS IS THE FUNCTION TO USE IF WE NEEDED TO FILL ANY OF THESE MISSING DATA
df1.sort_values(["average_price", "year"], ascending=[True, False]) #This sorts values based on the average price (ascending) & year (decending)
df1[df1["average_price"]<1] # This will print rows where the [average_price] is less than 1
df1[df1["type"] =="organic"]