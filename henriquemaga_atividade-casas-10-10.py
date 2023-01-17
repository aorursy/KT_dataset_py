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
casas = pd.read_csv('/kaggle/input/housedata/data.csv')                               
import statistics as stt

casas.describe()
casas['price'].min()

casas['price'].max()
casas['bathrooms'].max()
casas['waterfront'].sum()
casas['city'].nunique()

casas['city'].value_counts().max()
casas['city'].value_counts()
casas['city'][casas['city'].value_counts().max()]


import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

plt.hist(casas['city'].value_counts())

plt.grid()



plt.show()
casas.corr()
import matplotlib.pyplot as plt

plt.pie(casas['floors'].value_counts(), labels=(1,2,3,4,5,6))