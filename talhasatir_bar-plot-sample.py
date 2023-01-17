# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

all_data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
all_data.head()
all_data.info()
all_data.Rating.value_counts()
all_data["Rating"].fillna(all_data.groupby("Category")["Rating"].transform("mean"), inplace=True)
all_data.columns
all_data['Category'].unique()
area_list=list(all_data['Category'].unique())

rating_ratio=[]

for i in area_list:

    x=all_data[all_data['Category']==i]#x'i categorideki degerlerden birine eşitledim

    rating_rate=sum(x.Rating)/len(x)#categorideki degerle aynı satırı paylasan tüm Rating kolonundaki degerleri toplayıp ,sayısına bölüp ortalammalarını buldum

    rating_ratio.append(rating_rate)#bu ortalamayı categorik degerle aynı indis olmak sartı ile rating_ratio dizisine atıyorum

data=pd.DataFrame({'area_list':area_list,'rating_ratio':rating_ratio})#dataframeyi olusturdum

new_index=(data['rating_ratio'].sort_values(ascending=False)).index.values#verileri artandan azalana sıralıyorum 

sorted_data=data.reindex(new_index)#=indexlerini güncelliyorum



#figure

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['rating_ratio'])#azalandan artacak sekilde indislere göre x,y eksenlerini yerlestridim

plt.xticks(rotation=90)

plt.xlabel('states')

plt.ylabel('uzayli')

plt.title('uzaydan saygılar')
plt.figure(figsize=(15,10))

ax=sns.barplot(x=area_list,y=rating_ratio,palette=sns.cubehelix_palette(len(x)))

plt.xlabel('ss')

plt.ylabel('yy')

plt.title('dd')

plt.xticks(rotation=90)