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
data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

data.info()

data1=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

data1.info()
data.corr()

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths= .5, fmt= ' .1f', ax=ax)#heatmap=correlation map

#annot true olunca kutudaki degerleri gosteriyor

#.1f virgulden sonraki basamak sayisi icin

#ax=ax olmasi x ve y deki isimler karsilikli olsun diye

plt.show()

data1.corr()

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data1.corr(), annot=True, linewidths= .5, fmt= ' .1f', ax=ax)

plt.show()
data.head(10)

data1.head(10)
data.columns


data.Rating.plot(kind='line' , color = 'g', label = 'Rating' , linewidth=1, alpha= 0.5, grid = True, linestyle='-.')

plt.legend(loc='upper right')#legend = puts label into plot

plt.xlabel('x axis') 

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()

data.plot(kind = 'scatter', x = 'Rating', y= 'Reviews', alpha = 0.5, color = 'red')

plt.xlabel('Rating')

plt.ylabel('Reviews')

plt.title('Rating Reviews Scatter Plot')

plt.show()
data_frame = data[['Reviews']]  

print(data_frame)
x = data[['Reviews']]>2000000

data[x]

print(x)

print(data[x])
data1[(data1['Sentiment_Subjectivity']>0.5) & (data1['Sentiment_Polarity'],0.5)]