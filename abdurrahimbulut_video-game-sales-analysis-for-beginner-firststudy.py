# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
data.info()
data.head(10)
data.tail(10)
data.columns
data.Year_of_Release.plot(kind = 'hist',bins = 25,figsize = (9,9),label = 'Rating')

plt.legend()

plt.show()
from collections import Counter

counts = Counter(data.Genre)

df = pd.DataFrame.from_dict(counts, orient='index')

df.plot(kind='bar',figsize = (9,9))

plt.ylabel('frequency')            

plt.xlabel('Category')
data.plot(kind='scatter', x='Global_Sales', y='Year_of_Release',alpha = 0.5,color = 'red')

plt.xlabel('Global_Sales')    

plt.ylabel('Year_of_Release')

plt.title('Attack Defense Scatter Plot') 
from collections import Counter

counts = Counter(data.Platform)

df = pd.DataFrame.from_dict(counts, orient='index')

df.plot(kind='bar',figsize = (11,11))

plt.ylabel('frequency')            

plt.xlabel('Platform')
dictionary = {'maths' : 'Cahit Arf','chemistry' : 'Oktay Sinanoğlu'}

print(dictionary.keys())

print(dictionary.values())
dictionary['maths'] = "Ali Kuşçu"      # Update existing entry tr-Mevcut değeri güncellemek

print(dictionary)

dictionary['physics'] = "İsmail Akbay" # Add new entry tr-Yeni kayıt ekle

print(dictionary)

del dictionary['maths']                # remove entry with key 'maths' tr-key'i 'maths' olan kaydı sil

print(dictionary)

print('physics' in dictionary)          # check include or not tr-dictionary'e dahil olup olmadığını kontrol et

dictionary.clear()                     # remove all entries in dict  tr-dictionary değerlerini sil

print(dictionary)
# del dictionary         # delete entire dictionary tr- dictionary'i tamamen hafızadan siler

print(dictionary)  
data=pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
series = data['Developer']        # data['Developer'] = series

print(type(series))

data_frame = data[['Developer']]  # data[['Developer']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['Year_of_Release']>2016     # There are only 4 game who have higher year of release than 2016

data[x]
# 2 - Filtering pandas with logical_and

# There are only 4 game who have higher year of release than 2010 and global sales value than 12M

data[np.logical_and(data['Year_of_Release']>2010, data['Global_Sales']>12 )]