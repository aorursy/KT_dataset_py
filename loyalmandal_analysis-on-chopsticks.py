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
import os

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
data = pd.read_csv("../input/chopsticks-1992/chopstick-effectiveness.csv")
data.head(10)
data.describe()
data.info()
data = data.drop('Individual', axis = 1)
data.head()
data.rename(columns = {'Food.Pinching.Efficiency': 'efficiency', 'Chopstick.Length': 'len_chop'}, inplace=True)

data.head()
plt.plot(data['len_chop'], data['efficiency'])

plt.xlabel("Length")

plt.ylabel("Efficiency")

plt.show()



#Highest efficiency is found when length is 240
# There are six length

plt.hist(data['len_chop'])
data.nunique()
plt.hist(data['efficiency'])

plt.show()
# Check correlation between features

sns.pairplot(data)


