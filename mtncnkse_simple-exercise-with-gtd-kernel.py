# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  #visualization tool
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# global terrorism data and encoding (UniCodeDecode to not give error)

data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding= 'ISO-8859-1')
# check content and columns name of the data

data.head(20)
# correlation for each columns
data.corr()
# to compare Turkey and U.S.A
data_tr = data[data.country_txt =='Turkey']
data_us = data[data.country_txt == 'United States']
data_tr.nkill.plot(kind = 'line', grid= True, label= 'nkill', color = 'r', linewidth = 1, alpha = 0.5, linestyle = ':' )
data_tr.nkillus.plot(kind = 'line', grid = True, label = 'nkillus',  color = 'g', linewidth = 1, alpha = 0.5, linestyle = '-.')
plt.legend(loc = 'upper right')
plt.title("to compare to be killed in U.S.a and Turkey")
plt.show()
data_tr.info()
data_us.info()
data_tr.plot(kind= 'scatter', x= 'nkill', y= 'nwound', grid= True, alpha = 0.5, color = 'blue')
plt.title('to compare between nkill and nwound with scatter')
plt.xlabel('nkill')
plt.ylabel('nwound')
plt.show()
data.region.plot(kind= ' hist ', bins = 50, figsize=(12, 12))
plt.show()
cities = data_tr.provstate.dropna(False)
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     repeat = False,       
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.imsave(arr = wordcloud, fname = 'wordcloud.png')
plt.show()


x_city = data_tr.provstate.unique()
y_city_count = data_tr.provstate.value_counts(dropna= False)
plt.subplots(figsize=(15,60))
sns.barplot(x= x_city, y= y_city_count)
plt.xticks(rotation = 90)
plt.xlabel('cities that are attacked')
plt.ylabel('the number of injuries and dead')
plt.savefig('Attack')
plt.show()



    
