# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_filepath = '../input/cost-of-living/cost-of-living.csv'

my_data = pd.read_csv(my_filepath, header=0, index_col=0)

my_data.head()
my_dataT = my_data.T

my_dataT.columns
plt.figure(figsize=(10,8))

plt.title('COST OF LIVING IN AND OUT OF THE CITY CENTER')

plt.ylabel('Cost per month ($)')

# Create a plot

sns.boxplot(data=my_dataT[['Apartment (1 bedroom) in City Centre','Apartment (1 bedroom) Outside of Centre']], linewidth=3)

plt.show()
plt.figure(figsize=(15,8))

plt.title('SALARY VS RENT')



# Create a plot

sns.regplot(data=my_dataT, x = 'Average Monthly Net Salary (After Tax)', y = 'Apartment (1 bedroom) in City Centre', color = 'black')

plt.show()