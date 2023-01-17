import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import ExcelWriter

from pandas import ExcelFile

import io

import seaborn as sns



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns







%matplotlib inline



# Any results you write to the current directory are saved as output.
plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize']=(10,10)
data=pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='latin1')

data.info()
data=pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding='latin1')

data.info()
data.head()
#using 'sum' function will show us how many nulls are found in each column in dataset

data.isnull().sum()
#ok no null values
#examining the unique values of n_group as this column will appear very handy for later analysis

data.state.unique()
#coming back to our dataset we can confirm our fidnings with already existing column called 'calculated_host_listings_count'

top_fire_state=data.number.max()

top_fire_state
#Lets see wich state contains more fire occourrences



ax = data.groupby(['state']).sum()['number'].sort_values(ascending=False).plot(kind='bar')



# Change the y axis label to Arial

ax.set_ylabel('Total', fontname='Arial',fontsize=12)



# Set the title to comic Sans

ax.set_title('states with the most fire occourrences',fontname='Comic Sans MS', fontsize=18)
#Lets see wich month contains more fire occourrences



plt.figure(figsize=(15,7))

sns.boxplot(x='month',y='number',data=data[['month','number']])
# Number of fires per Year

plt.figure(figsize=(20,7))

sns.boxplot(x='year',y='number',data=data[['year','number']])