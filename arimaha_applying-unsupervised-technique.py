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


import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



data = pd.read_csv('../input/dataset.csv')



print(data.info()), print(data.shape)

# As we can see below there are two object type columns, 

# and the price is one of them, 

# nevertheless the price should be float object



# there are 43 feature columns (after excluding the ID column)

# and 337 observations



data.sum(axis=0)

# the summetion of values for each column,

# here we notice that 7 columns summetion values are = 0 

# which means that there is no observation related to this column 



print("1- Material per Item \n", data.mat_sum.describe(), "\n") 

print("2- Crystal per Item \n", data.cry_sum.describe(), "\n") 

print("3- Gemstone per Item \n", data.gem_sum.describe(), "\n")

# these three columns shows the different mix in one item

# such as different material in one item (gold, silver,.. etc)

# and different gemstones in one item  (pearl, opal,.. etc)

# and different crystals (dimond or ruby)



data.head()

# notice here that each time a earring flag is observed 

# it is compined with ring flag, this is due to the method used 

# to detect the item kind, using string methods to detect a substring

# and the word earring containes the word ring



pd.DataFrame(data.sum(axis=0)[4:], columns=['values summetion']).sort_values(by='values summetion').plot(kind='barh',figsize=(15,10))

# notice that the ring column is the sum of rings items and earrings 

# that is why it shows almost douple the amount of the other kind items



sns.set(rc={'figure.figsize':(15,15)})

mask = np.zeros_like(data.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    sns.heatmap(data.corr(), center=0, mask=mask, square=True)

    

# This heat map shows the correlation between features 

# there is some intresting positive and negative relations

# and the best_seller items dose not show strong correlation 

# and the white stribes are for where the columns have no values at all 

    


# 1- Changing the price column type to become numaric

## a- remove SR and comma

## b- convert to float 

data['price']= data['price'].map(lambda x : x.replace('SR','').replace(',',''))

data['price'] = pd.to_numeric(data['price'])

# print(data['price']);



# 2- Delete the unnessessarly columns 

for column in data.columns:

    if data[column].sum(axis=0)==0:

        data.pop(column)

data.sum(axis=0), data.shape



## inspecting the data correlation again 

sns.set(rc={'figure.figsize':(15,15)})

mask = np.zeros_like(data.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    sns.heatmap(data.corr(), center=0, mask=mask, square=True)

# There is strong correlation between the ring and the earring 

# becuse the word earring contains the word ring 



# Remove the correlation between ring and earring 

for i in range(len(data)):

    if data.earring.iloc[i]==1:

        data.ring.iloc[i]=0

        


## inspecting the data correlation again 

sns.set(rc={'figure.figsize':(15,15)})

mask = np.zeros_like(data.corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    sns.heatmap(data.corr(), center=0, mask=mask, square=True)

   


## inspecting again 

pd.DataFrame(data.sum(axis=0)[4:], columns=['values summetion']).sort_values(by='values summetion').plot(kind='barh',figsize=(15,10))

# notice how the ring moved from the top 5 



data.plot(x='price', y= 'mat_sum', kind='scatter', figsize=(8,5))

# the range of price reatches to 12K riyal 



data.plot(x='price', y= 'gem_sum', kind='scatter', figsize=(8,5))



data.plot(x='price', y= 'cry_sum', kind='scatter', figsize=(8,5))

data.describe()


# the range of prices and thier freuenty

pd.DataFrame(data['price'].value_counts()).sort_values(by='price').plot(kind='bar', figsize=(17,5))



# each observation price assossiated by the index 

pd.DataFrame(data['price']).sort_values(by='price').plot(kind='barh', figsize=(7,70)

                                                        )