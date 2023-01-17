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

import pandas as pd

import os
file = pd.read_csv("../input/streamset/Test.csv")

file.head()
file.describe()
file.loc[0:5, 'Booking':'Total Time']

# some imports to set up plotting 

import matplotlib.pyplot as plt

# pip install seaborn 

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

from seaborn import countplot

from matplotlib.pyplot import figure, show
pd.crosstab(file['Billing State (abv)'], file['Stage'], margins=True)

import pandas as pd

file2 = pd.read_csv("../input/testing/Testing.csv")

file2.head()
pd.crosstab(file2['What killed the deal'], file2['Stage'], margins=True)

pd.crosstab(file2['Last Lead Source'], file2['Stage'], margins=True)

pd.crosstab(file2['Total Time'], file2['Stage'], margins=True)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd
city_count  = file2['Billing State (abv)'].value_counts()

city_count = city_count[:10,]

plt.figure(figsize=(20,10))

sns.barplot(city_count.index, city_count.values, alpha=0.5)

plt.title('Analysis')

plt.ylabel('Total Number', fontsize=12)

plt.xlabel('Billing State (abv)', fontsize=8)

plt.show()
city_count  = file2['What killed the deal'].value_counts()

city_count = city_count[:10,]

plt.figure(figsize=(20,10))

sns.barplot(city_count.index, city_count.values, alpha=0.5)

plt.title('Analysis')

plt.ylabel('Total Number', fontsize=12)

plt.xlabel('What killed the deal ', fontsize=6)

plt.show()
city_count  = file2['Stage'].value_counts()

city_count = city_count[:10,]

plt.figure(figsize=(20,10))

sns.barplot(city_count.index, city_count.values, alpha=0.5)

plt.title('Analysis')

plt.ylabel('Total Number', fontsize=12)

plt.xlabel('Stage', fontsize=6)

plt.show()