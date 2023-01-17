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
#Importing required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Reading the csv file using pandas

data = pd.read_csv(r'/kaggle/input/most-popular-programming-languages-since-2004/Most Popular Programming Languages from 2004 to 2020.csv')



#Convering the date to Datetime data type

data['Date'] = pd.to_datetime(data['Date'])



#Setting date as index

data.set_index('Date', inplace = True)



#Displying first few lines of the dataset

data.head()
import missingno as miss



miss.matrix(data)
data.describe()
#DataFrame Information

data.info()
mask = data.mean() > 2.5



data = data.loc[:, mask]



clms = data.columns.tolist()

clms
#%matplotlib notebook

#pd.plotting.register_matplotlib_converters()

 

plt.figure(figsize = (15, 8))

sns.set(style = 'dark')



for language in clms:

    sns.lineplot(x = data.index, y = data[language], label = language)



plt.ylabel('Popularity', fontsize = 12)

plt.xlabel('Year', fontsize = 12)

plt.title('Popular Prgramming Languages', fontsize = 20)

plt.legend(loc = 2)

plt.yticks(fontsize = 10)

plt.xticks(rotation = 45, fontsize = 10)

plt.tight_layout()

plt.show()