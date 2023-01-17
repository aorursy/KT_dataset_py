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
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head()
df.info()
negara = df.groupby(['country'])['country'].count()

negaraa = negara[:10][::-1]

plt.barh(negaraa.index,negaraa)

plt.xlabel('Jumlah')

plt.title('Penayangan 10 Negara urut berdasarkan abjad')

plt.show()
tipe = df.groupby(['type'])['show_id'].count()

plt.bar(tipe.index,tipe)

plt.ylabel('Jumlah')

plt.title('Tipe')

plt.show()
rilis = df.release_year.value_counts()

plt.bar(rilis.index,rilis)

plt.ylabel('Jumlah')

plt.title('Perilisan')

plt.show()
rating = df.groupby(['rating'])['show_id'].count()

plt.barh(rating.index,rating)

plt.xlabel('Jumlah')

plt.title('Rating')

plt.show()
genre = df.groupby(['listed_in',])['listed_in'].count()

genre

#plt.bar(genre.index,genre)

#plt.ylabel('Jumlah')

#plt.title('Genre')

#plt.show()
negara=df.country.value_counts()



negaraa = negara[:10][::-1]

plt.barh(negaraa.index,negaraa)

plt.xlabel('Jumlah')

plt.title('10 Negara dengan konten terbanyak')

plt.show()

aa = df.loc[df['type'] == 'Movie']

aa

#top10 = df['country'].value_counts().nlargest(10)

#df = df[df['country'].isin(top10.index)]
