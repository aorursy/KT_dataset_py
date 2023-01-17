# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

df.info() #datasette bulunan değerler hakkında bilgi ediniyorum 
df.head() #ilk 5 satırını görebiliyoruz. Data hakkında genel fikir ediniyoruz. 
#Price ile user rating sayısı arasında ters koralasyon bekliyorum. Ücretli applere ilgi az olacağını düşünerek. 

df.corr()
#korelasyon tablosu zaten küçük ama ısı haritası

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap (df.corr(), annot = True,ax= ax)

plt.show()

#korelasyonu ısı haritası şeklinde gösteriyor. hiç bir veri arasında güçlü bir korelasyon yok. 
#Kolon isimlerim boşluklu olduğu için yeniden isimlendirdim. 

df.rename(columns={'Average User Rating': 'Avg_Rating', 'User Rating Count': 'Rating_Count'}, inplace=True)

df.columns

df.Price.plot(kind= 'line', color='r', label = 'Price', linewidth = 1, grid= True, linestyle= ':' )

plt.legend(loc = 'upper right')

plt.show()
#korelasyon olmadığı için scatter yapamıyorum. histogram ile puanlamanın nasıl dağıldığını göreceğim. 

df.Avg_Rating.plot(kind='hist', color='red', bins=50)

#puanlaması 5 ve ücretsiz olan appleri görmek istiyorum. 



filter1 = df['Avg_Rating'] = 5

filter2 = df['Price'] ==0

df[filter1 & filter2]




