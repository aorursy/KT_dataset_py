# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
data.info()
data.corr()
print(type(data.date_added[1]))

data.date_added=pd.to_datetime(data.date_added)

print(type(data.date_added[1]))
data.head()
en_cok_filmi_olan_ulke = data.country.value_counts()[:10]

pd.Series(en_cok_filmi_olan_ulke).sort_values(ascending=True).plot.barh(width=0.9, color=sns.color_palette('ocean_r', 10))

plt.title("Netflix' de en çok filmi olan 10 ülke")

plt.show()
labels = 'Film', 'Dizi'

sizes = data['type'].value_counts()

colors = ["lightpink", "yellowgreen"]

explode=(0.05,0) # chartın 1. parçasını 2. parçadan 0.05 kırdık

plt.pie(sizes,colors=colors,explode=explode, autopct='%1.1f%%',shadow=True) # shadow ile chartımızı gölgelendirdik

plt.legend(labels,loc='upper right') # loc ile labellarımızı sağ üst köşeye yerleştirdik

plt.show()
# İçeriklerin eklendiği yılları, ayrı bir kolon olarak ekliyoruz.

data['year']=data['date_added'].apply(lambda x: x.year)



data_movie=data[data['type']=='Movie'] # type ı Movie olanlar

data_tv_show=data[data['type']=='TV Show'] # type ı Movie olanlar

## density plot

sns.distplot(data_movie.year,hist=False, color="r",label="Film")

sns.distplot(data_tv_show.year, hist=False,color="b",label="Dizi")

plt.title('Yıllara göre içerik artışı')

plt.xlabel("Yıl")

plt.show()
data.rating.value_counts().plot(kind='bar',figsize=(14,8),title='Film ve dizilerin reytingleri',color='m')

plt.xlabel('Reyting Adları')

plt.ylabel('Reyting Sayıları')

plt.show()