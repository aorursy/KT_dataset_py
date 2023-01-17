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
data = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
data.info()
data.corr()
f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(data.corr(), annot=True, linewidths=.7, fmt= '.1f',ax=ax)

plt.show()
data.head(8)
data.columns
data.text_reviews_count.plot(kind = 'line', color = 'g',label = 'Text Reviews Count',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')

data.ratings_count.plot(color = 'r',label = 'Rating Count',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')    

plt.xlabel('x axis')            

plt.ylabel('y axis')

plt.title('Text Reviews Count & Rating Count')          

plt.show()
data.plot(kind='scatter', x='average_rating', y='# num_pages',alpha = 0.5,color = 'red')

plt.xlabel('Average Rating')             

plt.ylabel('Number Of Pages')

plt.title('Scatter Plot')           
data.average_rating.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
x=data['# num_pages']<200

data[x].count
data[x].shape
data.columns
data[np.logical_and(data['average_rating']>4.8, data['# num_pages']<100)]
data[(data['language_code']=='spa') & (data['text_reviews_count']>450)]
y=data['authors']=='José Saramago'

data[y]
i = 0

while i != 7 :

    print(i)

    i +=1 

print('Finish')

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')
a=data['language_code']=='tur'

data[a]
data.groupby(['language_code']).mean()