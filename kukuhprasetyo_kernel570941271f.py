# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import matplotlib.pyplot as plt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.head()



# Any results you write to the current directory are saved as output.
#Berikut data mina genre dan penjualan di negara ini

df.plot(x='Genre', y='NA_Sales', style='o')

plt.title('Genre and NA Sales')

plt.xlabel('Genre')

plt.ylabel('NA_Sales')

plt.show()

#dari data tersebut dapat di simpulkan bahwa North America genre game yang paling di minati adalah genre sport
#Berikut data mina genre dan penjualan di negara ini

df.plot(x='Genre', y='EU_Sales', style='o')

plt.title('Genre and EU Sales')

plt.xlabel('Genre')

plt.ylabel('EU_Sales')

plt.show()

#dari data tersebut dapat di simpulkan bahwa Britania Raya genre game yang paling di minati adalah genre sport
#Berikut data mina genre dan penjualan di negara ini

df.plot(x='Genre', y='JP_Sales', style='o')

plt.title('Genre and JP Sales')

plt.xlabel('Genre')

plt.ylabel('JP_Sales')

plt.show()

#dari data tersebut dapat di simpulkan bahwa Jepang genre game yang paling di minati adalah genre sport
#Berikut data mina genre dan penjualan di negara ini

df.plot(x='Genre', y='Other_Sales', style='o')

plt.title('Genre and Other Sales')

plt.xlabel('Genre')

plt.ylabel('Other_Sales')

plt.show()

#dari data tersebut dapat di simpulkan bahwa di negara lainya juga genre game yang paling di minati adalah genre sport
#Berikut data mina genre dan penjualan di negara ini

df.plot(x='Genre', y='Global_Sales', style='o')

plt.title('Genre and Global Sales')

plt.xlabel('Genre')

plt.ylabel('Global_Sales')

plt.show()

#dari data tersebut dapat di simpulkan bahwa secara global genre game yang paling di minati adalah genre sport
#ini adalah diagram antar rank dan genre

df.plot(x='Genre', y='Rank', style='o')

plt.title('Genre and Rank')

plt.xlabel('Genre')

plt.ylabel('Rank')

plt.show()

#di mana di dapat hasil jika rank no 1 adalah masih genre sport dan di ikuti dengan genre racing
#ini adalah diagram antar rank dan genre

df.plot(x='Genre', y='Rank', style='o')

plt.title('Genre and Rank')

plt.xlabel('Genre')

plt.ylabel('Rank')

plt.show()

#di mana di dapat hasil jika rank no 1 adalah masih genre sport dan di ikuti dengan genre racing