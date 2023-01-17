# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/books.csv', error_bad_lines=False)

data.head(10)
data.describe()
data.info()
data.language_code.unique()
sns.relplot(x='language_code',y='# num_pages',kind='line',data=data)

plt.xticks(rotation=90)

plt.show()
sns.catplot(x='language_code',y='average_rating',data=data)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

sns.barplot(x='language_code',y='average_rating',data=data)

plt.show()
f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='language_code',y='# num_pages',data=data,color='lime',alpha=0.8)

sns.pointplot(x='language_code',y='text_reviews_count',data=data,color='red',alpha=0.8)

plt.text(30,0.65,'text review',color='red',fontsize = 17,style = 'italic')

plt.text(33,0.50,'num page',color='lime',fontsize = 18,style = 'italic')



plt.xlabel('Languages',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.grid()
sns.jointplot("average_rating", "# num_pages", data=data,size=5, ratio=3, color="r")